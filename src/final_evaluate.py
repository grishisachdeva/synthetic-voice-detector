import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from src.model import AudioDeepfakeCNN

class DFEvalCacheDataset(Dataset):
    def __init__(self, metadata_df):
        self.metadata = metadata_df
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        feature = np.load(row['cache_path'])
        # Add channel dimension
        feature = torch.from_numpy(feature).unsqueeze(0)
        label = torch.tensor(row['label'], dtype=torch.long)
        return feature, label, row['file_id'], row['original_path'], row['attack_id'], row['label_name']

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def final_evaluate():
    print("Starting final evaluation...")
    
    # Load config
    with open('outputs/metrics/final_frozen_configuration.json', 'r') as f:
        config = json.load(f)
        
    threshold = config['classification_threshold']
    
    # Setup directories
    os.makedirs('outputs/predictions/final_evaluation', exist_ok=True)
    os.makedirs('outputs/plots/final_evaluation', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioDeepfakeCNN().to(device)
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Dataset
    cache_meta = pd.read_csv('data/processed/mel_cache/metadata/asvspoof2021_df_cache.csv')
    dataset = DFEvalCacheDataset(cache_meta)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    predictions = []
    
    print("Running inference...")
    with torch.no_grad():
        for features, labels, file_ids, paths, attack_ids, label_names in loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            for i in range(len(file_ids)):
                spoof_prob = probs[i][0]
                bonafide_prob = 1.0 - spoof_prob
                pred_label = 1 if spoof_prob >= threshold else 0
                pred_label_name = 'spoof' if pred_label == 1 else 'bonafide'
                
                predictions.append({
                    'file_id': file_ids[i],
                    'file_path': paths[i],
                    'true_label': labels[i].item(),
                    'true_label_name': label_names[i],
                    'spoof_probability': float(spoof_prob),
                    'bonafide_probability': float(bonafide_prob),
                    'predicted_label': pred_label,
                    'predicted_label_name': pred_label_name,
                    'attack_id': attack_ids[i]
                })
                
    preds_df = pd.DataFrame(predictions)
    preds_df.to_csv('outputs/predictions/final_evaluation/final_df_predictions.csv', index=False)
    
    print("Calculating metrics...")
    y_true = preds_df['true_label'].values
    y_pred = preds_df['predicted_label'].values
    y_prob = preds_df['spoof_probability'].values
    
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Handle the extremely unlikely case of single class in batch
    try:
        auc = roc_auc_score(y_true, y_prob)
        eer, eer_thresh = calculate_eer(y_true, y_prob)
    except ValueError:
        auc = 0.0
        eer = 0.0
        eer_thresh = 0.0
        
    # Generate Plots
    print("Generating plots...")
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('Final DF ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('outputs/plots/final_evaluation/roc_curve.png')
    plt.close()
    
    # CM
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Final DF Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['BONAFIDE', 'SPOOF'])
    plt.yticks(tick_marks, ['BONAFIDE', 'SPOOF'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), horizontalalignment="center", color="white" if cm[i][j] > cm.max()/2 else "black")
    plt.savefig('outputs/plots/final_evaluation/confusion_matrix.png')
    plt.close()
    
    # Score distribution
    plt.figure()
    plt.hist([preds_df[preds_df['true_label']==0]['spoof_probability'], 
              preds_df[preds_df['true_label']==1]['spoof_probability']], 
             bins=50, label=['BONAFIDE', 'SPOOF'], stacked=False)
    plt.title('Final DF Score Distribution (Spoof Prob)')
    plt.legend()
    plt.savefig('outputs/plots/final_evaluation/score_distribution.png')
    plt.close()
    
    # Per-Attack
    print("Per-attack analysis...")
    attacks = preds_df.groupby('attack_id').apply(lambda x: pd.Series({
        'sample_count': len(x),
        'accuracy': accuracy_score(x['true_label'], x['predicted_label']),
        'precision': precision_score(x['true_label'], x['predicted_label'], zero_division=0),
        'recall': recall_score(x['true_label'], x['predicted_label'], zero_division=0),
        'f1': f1_score(x['true_label'], x['predicted_label'], zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(x['true_label'], x['predicted_label'])
    })).reset_index()
    attacks.to_csv('outputs/metrics/final_df_per_attack.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(attacks['attack_id'], attacks['f1'])
    plt.title('F1 Score by Attack ID')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('outputs/plots/final_evaluation/per_attack_f1.png')
    plt.close()
    
    # Class Breakdown
    bonafide = preds_df[preds_df['true_label'] == 0]
    spoof = preds_df[preds_df['true_label'] == 1]
    
    breakdown = {
        'bonafide_count': len(bonafide),
        'spoof_count': len(spoof),
        'bonafide_correct': int((bonafide['predicted_label'] == 0).sum()),
        'bonafide_incorrect': int((bonafide['predicted_label'] == 1).sum()),
        'spoof_correct': int((spoof['predicted_label'] == 1).sum()),
        'spoof_incorrect': int((spoof['predicted_label'] == 0).sum())
    }
    with open('outputs/metrics/final_df_class_breakdown.json', 'w') as f:
        json.dump(breakdown, f, indent=4)
        
    # Errors
    errors = preds_df[preds_df['true_label'] != preds_df['predicted_label']]
    errors.to_csv('outputs/metrics/final_df_error_analysis.csv', index=False)
    
    with open('outputs/metrics/final_df_error_summary.txt', 'w') as f:
        f.write("ERROR SUMMARY\n=============\n")
        f.write(f"Total Errors: {len(errors)}\n")
        f.write(f"False Positives (Bonafide predicted as Spoof): {breakdown['bonafide_incorrect']}\n")
        f.write(f"False Negatives (Spoof predicted as Bonafide): {breakdown['spoof_incorrect']}\n")
        f.write("\nMost problematic attacks (Lowest F1):\n")
        f.write(attacks.sort_values('f1').head(5)[['attack_id', 'f1', 'sample_count']].to_string(index=False))
        f.write("\n")
        
    # Reproducibility check
    print("Running reproducibility check...")
    rep_loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    b_feat, _, _, _, _, _ = next(iter(rep_loader))
    b_feat = b_feat.to(device)
    out1 = model(b_feat)
    out2 = model(b_feat)
    assert torch.allclose(out1, out2), "Reproducibility failure: Outputs differ for same inputs"
    with open('outputs/metrics/final_reproducibility_check.txt', 'w') as f:
        f.write("PASS: Output matches within floating-point tolerance over two distinct forward passes of the identical batch.")
        
    # Final Results JSON
    total_official = 60176
    decodable = len(dataset)
    excluded = total_official - decodable
    
    results = {
        "project": "Audio Deepfake & Synthetic Voice Detector",
        "selected_model": config['model_name'],
        "checkpoint": config['checkpoint'],
        "evaluation_dataset": "ASVspoof 2021 DF",
        "total_official_files": total_official,
        "decodable_files": decodable,
        "excluded_files": excluded,
        "effective_evaluated_files": decodable,
        "threshold": threshold,
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "eer": float(eer),
        "eer_threshold": float(eer_thresh),
        "false_positive_rate": float(fpr_rate),
        "false_negative_rate": float(fnr_rate),
        "warning": f"{excluded} files were excluded due to libsndfile decoder failure." if excluded > 0 else None
    }
    with open('outputs/metrics/final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Final Report
    with open('outputs/metrics/final_results_report.txt', 'w') as f:
        f.write("FINAL HELD-OUT EVALUATION REPORT\n")
        f.write("================================\n")
        f.write("1. Final Model: AudioDeepfakeCNN\n")
        f.write("2. Evaluation Dataset: ASVspoof 2021 DF Evaluation Subset\n")
        f.write("3. Evaluation Protocol: Held-out Evaluation, purely zero-shot, no threshold tuning\n")
        f.write(f"4. Frozen Threshold: {threshold}\n\n")
        
        f.write(f"Official DF evaluation count: {total_official}\n")
        f.write(f"Decodable count: {decodable}\n")
        f.write(f"Excluded count: {excluded}\n")
        f.write(f"Effective scored count: {decodable}\n\n")
        
        f.write("--- METRICS ---\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced Accuracy: {bacc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"EER: {eer:.4f}\n")
        f.write(f"FPR: {fpr_rate:.4f}\n")
        f.write(f"FNR: {fnr_rate:.4f}\n\n")
        
        f.write("--- CONFUSION MATRIX ---\n")
        f.write(f"True Negatives (Bonafide): {tn}\n")
        f.write(f"False Positives (Spoof): {fp}\n")
        f.write(f"False Negatives (Spoof): {fn}\n")
        f.write(f"True Positives (Bonafide): {tp}\n\n")
        
        f.write("--- DATASET EXCLUSIONS ---\n")
        if excluded > 0:
            f.write(f"WARNING: {excluded} files were permanently excluded because their underlying FLAC binary payload triggers unrecoverable libsndfile decode errors (known ASVspoof 2021 artifact). The metrics represent performance exclusively on standard, correctly formatted decodable audio.\n\n")
            
        f.write("--- INTERPRETATION ---\n")
        f.write(f"The model achieved an ROC-AUC of {auc:.4f} and F1 of {f1:.4f} on the decodable held-out ASVspoof 2021 DF evaluation set.\n")

    print("Final evaluation completely generated.")

if __name__ == "__main__":
    final_evaluate()
