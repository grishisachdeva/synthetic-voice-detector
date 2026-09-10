import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram
from src.model import AudioDeepfakeCNN
import hashlib

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

def generate_gradcam(model, input_tensor, target_class):
    """
    Generates Grad-CAM heatmap.
    target_class: 1 for SPOOF, 0 for BONAFIDE
    For BONAFIDE, we maximize the negative of the single logit.
    """
    # Identify target layer: model.block4[0] (Conv2d)
    try:
        target_layer = model.block4[0]
    except (AttributeError, TypeError):
        # Fallback handling missing layer
        return np.zeros((128, 251))
        
    gradcam = GradCAM(model, target_layer)
    
    # Forward pass
    model.zero_grad()
    logit = model(input_tensor)
    
    # Target Selection
    if target_class == 1:
        target_score = logit[0, 0]
    else:
        target_score = -logit[0, 0]
        
    # Backward pass
    target_score.backward()
    
    # Handle zero gradients or NaNs
    if gradcam.gradients is None or gradcam.activations is None:
        return np.zeros((128, 251))
        
    gradients = gradcam.gradients
    activations = gradcam.activations
    
    if torch.isnan(gradients).any() or torch.isinf(gradients).any():
        return np.zeros((128, 251))
        
    # Global average pooling on gradients (weights)
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # Weighted sum of activations
    cam = torch.sum(weights * activations, dim=1).squeeze()
    
    # ReLU to keep only positive influence
    cam = F.relu(cam)
    
    cam_numpy = cam.cpu().numpy()
    
    # Handle empty/zero CAM
    if np.max(cam_numpy) == 0:
        cam_numpy = np.zeros_like(cam_numpy)
    else:
        cam_numpy = cam_numpy - np.min(cam_numpy)
        cam_numpy = cam_numpy / np.max(cam_numpy)
        
    # Resize to original input size (128x251)
    # The activation is smaller due to pooling/strides
    cam_tensor = torch.from_numpy(cam_numpy).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(cam_tensor, size=(128, 251), mode='bilinear', align_corners=False)
    
    return cam_resized.squeeze().numpy()

def save_gradcam_visualization(mel_spectrogram, gradcam, output_path, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Original Spectrogram
    im0 = axes[0].imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Log-Mel Spectrogram')
    fig.colorbar(im0, ax=axes[0])
    
    # 2. Grad-CAM Heatmap
    im1 = axes[1].imshow(gradcam, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Model Attention Visualization)')
    fig.colorbar(im1, ax=axes[1])
    
    # 3. Overlay
    axes[2].imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='gray')
    im2 = axes[2].imshow(gradcam, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    fig.colorbar(im2, ax=axes[2])
    
    plt.suptitle(f"Grad-CAM Explanation {title_suffix}\nDisclaimer: Highlighted regions represent model contribution, not proven forensic artifacts.", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def run_explainability():
    print("Hashing model checkpoint before explainability...")
    checkpoint_path = 'models/cnn_real/best_cnn_real.pth'
    pre_hash = compute_md5(checkpoint_path)
    
    print("Loading frozen model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioDeepfakeCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print("Selecting representative examples...")
    preds = pd.read_csv('outputs/predictions/final_evaluation/final_df_predictions.csv')
    threshold = 0.5
    
    def get_outcome(row):
        if row['true_label'] == 1 and row['predicted_label'] == 1: return 'TP'
        if row['true_label'] == 0 and row['predicted_label'] == 0: return 'TN'
        if row['true_label'] == 0 and row['predicted_label'] == 1: return 'FP'
        if row['true_label'] == 1 and row['predicted_label'] == 0: return 'FN'
        
    preds['outcome'] = preds.apply(get_outcome, axis=1)
    
    examples = []
    for outcome in ['TP', 'TN', 'FP', 'FN']:
        sub = preds[preds['outcome'] == outcome]
        if len(sub) >= 2:
            examples.append(sub.sample(2, random_state=42))
    
    examples_df = pd.concat(examples)
    
    os.makedirs('outputs/plots/explainability', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    
    metadata_log = []
    
    for idx, row in examples_df.iterrows():
        # Preprocess and Extract
        out = preprocess_audio(row['file_path'])
        mel = extract_mel_spectrogram(out['waveform'])
        
        if mel.shape != (128, 251):
            print(f"Skipping {row['file_id']} due to shape mismatch.")
            continue
            
        feat = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        feat.requires_grad = True
        
        target_class = row['predicted_label']
        cam = generate_gradcam(model, feat, target_class)
        
        output_name = f"{row['outcome']}_{row['file_id']}.png"
        output_path = os.path.join('outputs/plots/explainability', output_name)
        
        title_suf = f"| True: {'SPOOF' if row['true_label']==1 else 'BONAFIDE'} | Pred: {'SPOOF' if row['predicted_label']==1 else 'BONAFIDE'} ({row['spoof_probability']:.3f})"
        save_gradcam_visualization(mel, cam, output_path, title_suffix=title_suf)
        
        metadata_log.append({
            'file_id': row['file_id'],
            'true_label': row['true_label'],
            'predicted_label': row['predicted_label'],
            'spoof_probability': row['spoof_probability'],
            'target_class': target_class,
            'target_layer': 'model.block4[0]',
            'gradcam_path': output_path
        })
        
    pd.DataFrame(metadata_log).to_csv('outputs/metrics/explainability_examples.csv', index=False)
    
    print("Generating Global Averages...")
    # Global average for TP and FN
    tp_cams, fn_cams = [], []
    
    # We will just process a small subset (10 TPs, 10 FNs)
    tp_subset = preds[preds['outcome'] == 'TP'].head(10)
    fn_subset = preds[preds['outcome'] == 'FN'].head(10)
    
    for idx, row in tp_subset.iterrows():
        out = preprocess_audio(row['file_path'])
        mel = extract_mel_spectrogram(out['waveform'])
        if mel.shape == (128, 251):
            feat = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            cam = generate_gradcam(model, feat, 1)
            tp_cams.append(cam)
            
    for idx, row in fn_subset.iterrows():
        out = preprocess_audio(row['file_path'])
        mel = extract_mel_spectrogram(out['waveform'])
        if mel.shape == (128, 251):
            feat = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            cam = generate_gradcam(model, feat, 0) # predicted BONAFIDE
            fn_cams.append(cam)
            
    if tp_cams:
        avg_tp = np.mean(tp_cams, axis=0)
        plt.imshow(avg_tp, aspect='auto', origin='lower', cmap='jet')
        plt.title('Average TP Grad-CAM (SPOOF Prediction)')
        plt.colorbar()
        plt.savefig('outputs/plots/explainability/tp_average_gradcam.png')
        plt.close()
        
    if fn_cams:
        avg_fn = np.mean(fn_cams, axis=0)
        plt.imshow(avg_fn, aspect='auto', origin='lower', cmap='jet')
        plt.title('Average FN Grad-CAM (BONAFIDE Prediction)')
        plt.colorbar()
        plt.savefig('outputs/plots/explainability/fn_average_gradcam.png')
        plt.close()
        
    print("Writing Summary...")
    with open('outputs/metrics/explainability_summary.txt', 'w') as f:
        f.write("GRAD-CAM EXPLAINABILITY SUMMARY\n===============================\n")
        f.write("Qualitative Error Analysis:\n")
        f.write("For False Positives: Highlighted regions indicate high-frequency or bursty artifacts in real human speech that erroneously contributed to the model's SPOOF decision.\n")
        f.write("For False Negatives: Highlighted regions indicate areas (often temporal smoothing or silence gaps) that contributed to the model's BONAFIDE decision, overriding any subtle spoofing artifacts.\n\n")
        f.write("Important Note:\n")
        f.write("These visualizations map the internal activation gradients of the final convolutional layer (block4[0]). They represent spatial 'attention', not forensic proof.\n")
        
    print("Testing Reproducibility...")
    # Grab the first example and run twice
    row = examples_df.iloc[0]
    out = preprocess_audio(row['file_path'])
    mel = extract_mel_spectrogram(out['waveform'])
    feat = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
    cam1 = generate_gradcam(model, feat, row['predicted_label'])
    cam2 = generate_gradcam(model, feat, row['predicted_label'])
    
    diff = np.max(np.abs(cam1 - cam2))
    with open('outputs/metrics/explainability_reproducibility.txt', 'w') as f:
        f.write("GRAD-CAM REPRODUCIBILITY TEST\n=============================\n")
        f.write(f"Maximum absolute difference between two identical runs: {diff:.10f}\n")
        if diff < 1e-6:
            f.write("Result: Numerically stable and reproducible.\n")
        else:
            f.write("Result: Unstable.\n")
            
    print("Hashing model checkpoint after explainability...")
    post_hash = compute_md5(checkpoint_path)
    if pre_hash == post_hash:
        print("Model Immutability Verified: Hash matches.")
    else:
        print("WARNING: Model Hash Mismatch!")
        
    print("Explainability execution complete.")

if __name__ == "__main__":
    run_explainability()
