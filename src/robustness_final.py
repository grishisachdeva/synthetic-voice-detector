import os
import torch
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, lfilter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram
from src.model import AudioDeepfakeCNN

def calculate_eer(y_true, y_score):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
    except ValueError:
        return 0.0

def add_gaussian_noise(waveform, snr_db):
    signal_power = np.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
    return waveform + noise

def change_gain(waveform, gain_db):
    return waveform * (10 ** (gain_db / 20))

def time_shift(waveform, shift_sec, sr=16000):
    shift_samples = int(shift_sec * sr)
    if shift_samples > 0:
        return np.pad(waveform, (shift_samples, 0))[:-shift_samples]
    elif shift_samples < 0:
        return np.pad(waveform, (0, -shift_samples))[-shift_samples:]
    return waveform

def butter_lowpass_filter(data, cutoff, sr=16000, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0: return data
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def run_robustness():
    print("Starting robustness testing...")
    
    # Load frozen model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioDeepfakeCNN().to(device)
    checkpoint = torch.load('models/cnn_real/best_cnn_real.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    threshold = 0.5
    
    # Load metadata and subset
    df = pd.read_csv('data/processed/mel_cache/metadata/asvspoof2021_df_cache.csv')
    
    # Select subset (1000 files)
    bonafide = df[df['label'] == 0].sample(n=500, random_state=42)
    spoof = df[df['label'] == 1].sample(n=500, random_state=42)
    subset = pd.concat([bonafide, spoof])
    
    conditions = [
        ('clean', None),
        ('gaussian_noise_20dB', lambda x: add_gaussian_noise(x, 20)),
        ('gaussian_noise_10dB', lambda x: add_gaussian_noise(x, 10)),
        ('gaussian_noise_5dB', lambda x: add_gaussian_noise(x, 5)),
        ('gain_variation_-6dB', lambda x: change_gain(x, -6)),
        ('gain_variation_-3dB', lambda x: change_gain(x, -3)),
        ('gain_variation_0dB', lambda x: change_gain(x, 0)),
        ('gain_variation_+3dB', lambda x: change_gain(x, 3)),
        ('gain_variation_+6dB', lambda x: change_gain(x, 6)),
        ('time_shift_-0.1s', lambda x: time_shift(x, -0.1)),
        ('time_shift_+0.1s', lambda x: time_shift(x, 0.1)),
        ('low_pass_8000Hz', lambda x: butter_lowpass_filter(x, 8000)),
        ('low_pass_4000Hz', lambda x: butter_lowpass_filter(x, 4000)),
        ('amplitude_scaling_x0.5', lambda x: x * 0.5),
        ('amplitude_scaling_x2.0', lambda x: x * 2.0),
        ('compression', None) # Unavailable
    ]
    
    results = []
    manifest = []
    
    for cond_name, transform_fn in conditions:
        print(f"Testing condition: {cond_name}")
        if cond_name == 'compression':
            results.append({
                'condition': cond_name,
                'sample_count': 'unavailable',
                'accuracy': 'unavailable', 'balanced_accuracy': 'unavailable',
                'precision': 'unavailable', 'recall': 'unavailable',
                'f1': 'unavailable', 'roc_auc': 'unavailable',
                'eer': 'unavailable', 'fpr': 'unavailable', 'fnr': 'unavailable'
            })
            continue
            
        y_true, y_pred, y_prob = [], [], []
        
        for idx, row in tqdm(subset.iterrows(), total=len(subset), leave=False):
            # Load raw audio
            waveform, sr = librosa.load(row['original_path'], sr=None)
            
            # Apply transform
            if transform_fn is not None:
                waveform = transform_fn(waveform)
                
            # Preprocess (trim/pad/normalize)
            # We mock the path behavior by saving to memory
            import soundfile as sf
            temp_path = "data/processed/temp_robustness.wav"
            sf.write(temp_path, waveform, sr, format='WAV')
            
            # Modified preprocess_audio locally to take buffer
            out = preprocess_audio(temp_path)
            mel = extract_mel_spectrogram(out['waveform'])
            
            if mel.shape != (128, 251):
                continue
                
            feat = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logit = model(feat)
                prob = float(torch.sigmoid(logit).cpu().numpy()[0][0])
                
            y_true.append(row['label'])
            y_prob.append(prob)
            y_pred.append(1 if prob >= threshold else 0)
            
            manifest.append({
                'source_file_id': row['file_id'],
                'condition': cond_name,
                'transformed_file_or_identifier': f"in_memory_{cond_name}"
            })
            
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        eer = calculate_eer(y_true, y_prob)
        
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        results.append({
            'condition': cond_name,
            'sample_count': len(y_true),
            'accuracy': acc, 'balanced_accuracy': bacc,
            'precision': prec, 'recall': rec,
            'f1': f1, 'roc_auc': auc,
            'eer': eer, 'fpr': fpr, 'fnr': fnr
        })
        
    os.makedirs('outputs/metrics', exist_ok=True)
    pd.DataFrame(results).to_csv('outputs/metrics/final_robustness_results.csv', index=False)
    pd.DataFrame(manifest).to_csv('outputs/metrics/robustness_sample_manifest.csv', index=False)
    
    # Plots
    print("Generating robustness plots...")
    rdf = pd.DataFrame([r for r in results if r['accuracy'] != 'unavailable'])
    os.makedirs('outputs/plots/final_evaluation', exist_ok=True)
    
    for metric in ['f1', 'recall', 'roc_auc', 'eer']:
        plt.figure(figsize=(12, 6))
        plt.bar(rdf['condition'], rdf[metric])
        plt.title(f'Robustness: {metric.upper()}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'outputs/plots/final_evaluation/robustness_{metric}.png')
        plt.close()
        
    print("Robustness complete.")

if __name__ == "__main__":
    run_robustness()
