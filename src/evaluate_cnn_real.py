import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

from src.model import AudioDeepfakeCNN
from src.cached_dataset import CachedMelSpectrogramDataset, create_cached_dataloaders
from src.config import BATCH_SIZE

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold

def evaluate(model, dataloader, device):
    model.eval()
    all_probs = []
    all_targets = []
    file_ids = []
    speaker_ids = []
    attack_ids = []
    file_paths = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(features).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())
            
            file_ids.extend(batch['file_id'])
            speaker_ids.extend(batch['speaker_id'])
            attack_ids.extend(batch['attack_id'])
            file_paths.extend(batch['path'])
            
    return np.array(all_probs), np.array(all_targets), file_ids, speaker_ids, attack_ids, file_paths

def plot_training_curves(history_csv, output_dir):
    df = pd.read_csv(history_csv)
    epochs = df['epoch']
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1-Score'),
    ]
    
    for col, title in metrics:
        plt.figure(figsize=(8, 6))
        if f'train_{col}' in df.columns:
            plt.plot(epochs, df[f'train_{col}'], label=f'Train {title}', color='blue')
        plt.plot(epochs, df[f'val_{col}'], label=f'Val {title}', color='orange')
        plt.title(f'Training vs Validation {title}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{col}_curve.png'))
        plt.close()
        
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['val_roc_auc'], label='Val ROC-AUC', color='purple')
    plt.title('Validation ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_auc_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Development Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['BONAFIDE (0)', 'SPOOF (1)'])
    plt.yticks(tick_marks, ['BONAFIDE (0)', 'SPOOF (1)'])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def do_overfitting_analysis(history_csv, output_path):
    df = pd.read_csv(history_csv)
    last_train_loss = df['train_loss'].iloc[-1]
    last_val_loss = df['val_loss'].iloc[-1]
    
    conclusion = "healthy convergence"
    if last_val_loss > last_train_loss * 1.5:
        conclusion = "overfitting"
    elif last_train_loss > 0.5 and last_val_loss > 0.5:
        conclusion = "underfitting"
        
    with open(output_path, "w") as f:
        f.write("OVERFITTING ANALYSIS\n")
        f.write(f"Final Train Loss: {last_train_loss:.4f}\n")
        f.write(f"Final Val Loss:   {last_val_loss:.4f}\n")
        f.write(f"Conclusion: {conclusion}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec-augment', action='store_true', help="Evaluate SpecAugment model instead of Unweighted")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    suffix = "_specaug" if args.spec_augment else ""
    model_path = f"models/cnn_real{suffix}/best_cnn_real.pth"
    history_file = f"outputs/metrics/cnn_real{suffix}_training_history.csv"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)
        
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = AudioDeepfakeCNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    dev_csv = "data/processed/mel_cache/metadata/development_cache.csv"
    dev_dataset = CachedMelSpectrogramDataset(dev_csv)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Evaluating development set...")
    probs, targets, file_ids, speaker_ids, attack_ids, file_paths = evaluate(model, dev_loader, device)
    
    # 1. Prediction CSV
    preds = (probs >= 0.5).astype(float)
    df_preds = pd.DataFrame({
        'file_id': file_ids,
        'file_path': file_paths,
        'speaker_id': speaker_ids,
        'attack_id': attack_ids,
        'true_label': targets,
        'true_label_name': ['spoof' if t == 1 else 'bonafide' for t in targets],
        'spoof_probability': probs,
        'bonafide_probability': 1.0 - probs,
        'predicted_label': preds,
        'predicted_label_name': ['spoof' if p == 1 else 'bonafide' for p in preds]
    })
    os.makedirs("outputs/predictions", exist_ok=True)
    df_preds.to_csv(f"outputs/predictions/cnn_real{suffix}_dev_predictions.csv", index=False)
    
    # 2. Confusion Matrix & Core Metrics (Threshold = 0.5)
    os.makedirs(f"outputs/plots/cnn_real{suffix}", exist_ok=True)
    plot_confusion_matrix(targets, preds, f"outputs/plots/cnn_real{suffix}/dev_confusion_matrix.png")
    
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2.0
    
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    roc_auc = roc_auc_score(targets, probs)
    eer, eer_threshold = compute_eer(targets, probs)
    
    metrics_json = {
        "dataset": "ASVspoof 2019 LA",
        "partition": "development",
        "model": "AudioDeepfakeCNN",
        "num_samples": len(targets),
        "num_bonafide": int(tn + fp),
        "num_spoof": int(tp + fn),
        "threshold": 0.5,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "specificity_tnr": specificity,
        "sensitivity_tpr": sensitivity,
        "class_weighting_used": config.get('weighted', False),
        "pos_weight": "BONAFIDE/SPOOF calculated" if config.get('weighted', False) else 1.0,
        "augmentation": "SpecAugment" if args.spec_augment else "None",
        "best_epoch": checkpoint.get('epoch', -1),
        "best_validation_metric": checkpoint.get('best_metric', -1)
    }
    with open(f"outputs/metrics/cnn_real{suffix}_dev_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=4)
        
    # 3. Threshold Analysis
    thresholds_to_test = np.arange(0.1, 0.95, 0.05)
    ta_data = []
    best_thresh_f1 = -1
    best_thresh = 0.5
    for th in thresholds_to_test:
        p_th = (probs >= th).astype(float)
        cm_th = confusion_matrix(targets, p_th)
        if len(cm_th.ravel()) == 4:
            tn_t, fp_t, fn_t, tp_t = cm_th.ravel()
            prec_t = precision_score(targets, p_th, zero_division=0)
            rec_t = recall_score(targets, p_th, zero_division=0)
            f1_t = f1_score(targets, p_th, zero_division=0)
            fpr_t = fp_t / (fp_t + tn_t)
            fnr_t = fn_t / (fn_t + tp_t)
            
            ta_data.append({
                "threshold": th, "precision": prec_t, "recall": rec_t, "f1": f1_t, "fpr": fpr_t, "fnr": fnr_t
            })
            if f1_t > best_thresh_f1:
                best_thresh_f1 = f1_t
                best_thresh = th
                
    pd.DataFrame(ta_data).to_csv(f"outputs/metrics/cnn_real{suffix}_dev_threshold_analysis.csv", index=False)
    
    with open(f"outputs/metrics/cnn_real{suffix}_selected_threshold.json", "w") as f:
        json.dump({"selected_threshold": best_thresh, "selection_metric": "F1", "selection_partition": "development"}, f, indent=4)
        
    # 4. Per-attack Analysis
    attack_data = []
    unique_attacks = sorted(list(set(attack_ids)))
    for atk in unique_attacks:
        if atk == "-":
            continue
        mask = (np.array(attack_ids) == atk) | (targets == 0) # Include all bonafide + this specific attack
        t_sub = targets[mask]
        p_sub = preds[mask]
        
        if len(np.unique(t_sub)) > 1:
            a_acc = accuracy_score(t_sub, p_sub)
            a_prec = precision_score(t_sub, p_sub, zero_division=0)
            a_rec = recall_score(t_sub, p_sub, zero_division=0)
            a_f1 = f1_score(t_sub, p_sub, zero_division=0)
            attack_data.append({"attack_id": atk, "sample_count": sum(np.array(attack_ids) == atk), "accuracy": a_acc, "precision": a_prec, "recall": a_rec, "f1": a_f1})
            
    pd.DataFrame(attack_data).to_csv(f"outputs/metrics/cnn_real{suffix}_dev_per_attack.csv", index=False)

    # 5. Per-speaker Analysis
    speaker_data = []
    unique_speakers = sorted(list(set(speaker_ids)))
    for spk in unique_speakers:
        mask = (np.array(speaker_ids) == spk)
        t_sub = targets[mask]
        p_sub = preds[mask]
        
        s_acc = accuracy_score(t_sub, p_sub)
        s_prec = precision_score(t_sub, p_sub, zero_division=0)
        s_rec = recall_score(t_sub, p_sub, zero_division=0)
        s_f1 = f1_score(t_sub, p_sub, zero_division=0)
        speaker_data.append({"speaker_id": spk, "sample_count": sum(mask), "accuracy": s_acc, "precision": s_prec, "recall": s_rec, "f1": s_f1})
        
    pd.DataFrame(speaker_data).to_csv(f"outputs/metrics/cnn_real{suffix}_dev_per_speaker.csv", index=False)
    
    # 6. Plotting
    if os.path.exists(history_file):
        plot_training_curves(history_file, f"outputs/plots/cnn_real{suffix}")
        do_overfitting_analysis(history_file, f"outputs/metrics/cnn_real{suffix}_overfitting_analysis.txt")
        
    print(f"Evaluation complete for Experiment {suffix if suffix else 'A'}.")
    print(f"EER: {eer:.4f} | ROC-AUC: {roc_auc:.4f} | Balanced Acc: {balanced_acc:.4f} | F1: {f1:.4f}")
    
if __name__ == "__main__":
    main()
