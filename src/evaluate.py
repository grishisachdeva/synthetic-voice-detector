# evaluate.py — Held-out evaluation pipeline for the trained CNN.

import os
import argparse
import json
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import CHECKPOINT_PATH, PREDICTION_THRESHOLD, BATCH_SIZE
from src.dataset import create_datasets, create_dataloaders
from src.model import create_model
from src.utils import calculate_metrics, load_checkpoint

def analyze_thresholds(y_true, probabilities, thresholds):
    """Analyze precision, recall, f1, fpr, fnr across multiple thresholds."""
    results = []
    y_true = np.array(y_true)
    probabilities = np.array(probabilities)
    
    for thresh in thresholds:
        preds = (probabilities >= thresh).astype(int)
        
        tp = np.sum((preds == 1) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = 2 * (precision * recall) / (precision + recall) if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0 else float('nan')
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')
        
        results.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr
        })
    return results

def calculate_eer(y_true, spoof_scores):
    """
    Calculate Equal Error Rate (EER) and the corresponding threshold.
    Returns (eer, eer_threshold).
    Uses NaN if both classes are not present or if the array is empty.
    """
    y_true = np.array(y_true)
    spoof_scores = np.array(spoof_scores)
    
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return float('nan'), float('nan')
        
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, spoof_scores, pos_label=1)
        fnr = 1 - tpr
        
        # Find intersection where FPR and FNR cross
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        eer_threshold = thresholds[idx]
        return eer, eer_threshold
    except Exception:
        return float('nan'), float('nan')

def plot_probability_distribution(y_true, spoof_scores, out_dir="outputs/plots"):
    """Plot distribution of spoof scores for BONAFIDE and SPOOF classes."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true)
    spoof_scores = np.array(spoof_scores)
    
    bonafide_scores = spoof_scores[y_true == 0]
    spoof_cases = spoof_scores[y_true == 1]
    
    plt.figure(figsize=(8, 5))
    if len(bonafide_scores) > 0:
        plt.hist(bonafide_scores, bins=10, alpha=0.5, label='BONAFIDE', density=False)
    if len(spoof_cases) > 0:
        plt.hist(spoof_cases, bins=10, alpha=0.5, label='SPOOF', density=False)
        
    plt.axvline(PREDICTION_THRESHOLD, color='k', linestyle='dashed', linewidth=1, label=f'Threshold ({PREDICTION_THRESHOLD})')
    plt.xlabel('Spoof Probability')
    plt.ylabel('Count')
    plt.title('Spoof Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, 'probability_distribution.png'), dpi=150)
    plt.close()

def plot_roc_curve(y_true, spoof_scores, out_dir="outputs/plots"):
    """Plot ROC curve."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true)
    spoof_scores = np.array(spoof_scores)
    
    if len(np.unique(y_true)) < 2:
        print("Skipping ROC curve plot: Evaluation set requires both classes.")
        return
        
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, spoof_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, 'test_roc_curve.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Failed to plot ROC curve: {e}")

def plot_confusion_matrix(y_true, spoof_scores, threshold=0.5, out_dir="outputs/plots"):
    """Plot confusion matrix using absolute counts."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true)
    preds = (np.array(spoof_scores) >= threshold).astype(int)
    
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BONAFIDE", "SPOOF"])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", values_format="d") # 'd' for absolute integer counts
        plt.title(f"Test Confusion Matrix (Threshold {threshold})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "test_confusion_matrix.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Failed to generate confusion matrix: {e}")

def evaluate_model(model_path, evaluation_loader, device):
    """
    Evaluate the saved model on the held-out evaluation dataset.
    Generates predictions and probabilities.
    """
    model = create_model()
    load_checkpoint(model_path, model)
    model.eval()
    
    results = []
    
    print("Running evaluation forward passes...")
    with torch.no_grad():
        for batch in evaluation_loader:
            features = batch["features"].to(device)
            labels = batch["label"].cpu().numpy()
            paths = batch["path"]
            attack_ids = batch.get("attack_id", ["-" for _ in range(len(paths))])
            
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            
            # Handle batch size 1 issue with squeeze
            if probs.ndim == 0:
                probs = np.expand_dims(probs, 0)
                
            for i in range(len(paths)):
                p_spoof = float(probs[i])
                p_bonafide = 1.0 - p_spoof
                t_label = int(labels[i])
                
                results.append({
                    "file_path": paths[i],
                    "true_label": t_label,
                    "true_label_name": "SPOOF" if t_label == 1 else "BONAFIDE",
                    "attack_id": attack_ids[i] if isinstance(attack_ids[i], str) else "-",
                    "spoof_probability": p_spoof,
                    "bonafide_probability": p_bonafide,
                    "predicted_label": 1 if p_spoof >= PREDICTION_THRESHOLD else 0,
                    "predicted_label_name": "SPOOF" if p_spoof >= PREDICTION_THRESHOLD else "BONAFIDE"
                })
                
    return results

def verify_data_leakage(train_ds, dev_ds, eval_ds):
    """Code-level check to ensure evaluation data did not leak into training."""
    train_paths = set(r["file_path"] for r in train_ds.metadata)
    dev_paths = set(r["file_path"] for r in dev_ds.metadata)
    eval_paths = set(r["file_path"] for r in eval_ds.metadata)
    
    train_overlap = train_paths.intersection(eval_paths)
    dev_overlap = dev_paths.intersection(eval_paths)
    
    if train_overlap or dev_overlap:
        raise ValueError("DATA LEAKAGE DETECTED! Evaluation samples found in training/development partitions.")
    print("Leakage Check: PASS (No evaluation paths found in train/dev metadata).")

def generate_classification_report(y_true, y_pred, out_path):
    """Generate textual classification report using sklearn."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=["BONAFIDE", "SPOOF"], labels=[0, 1], zero_division=0)
        with open(out_path, "w") as f:
            f.write("Evaluation Classification Report\n")
            f.write("==============================\n\n")
            f.write(report)
    except Exception as e:
        print(f"Failed to generate classification report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline CNN on Held-Out Data")
    parser.add_argument("--model-path", type=str, default=CHECKPOINT_PATH, help="Path to best CNN checkpoint")
    args = parser.parse_args()
    
    # 1. Leakage verification and Dataloader setup
    print("Initializing datasets...")
    train_ds, dev_ds, eval_ds = create_datasets()
    verify_data_leakage(train_ds, dev_ds, eval_ds)
    
    _, _, eval_loader = create_dataloaders(train_ds, dev_ds, eval_ds)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Evaluation Forward Passes
    results = evaluate_model(args.model_path, eval_loader, device)
    
    y_true = [r["true_label"] for r in results]
    probs = [r["spoof_probability"] for r in results]
    preds = [r["predicted_label"] for r in results]
    
    # 3. Save Predictions CSV
    pred_csv_path = "outputs/predictions/evaluation_predictions.csv"
    os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)
    keys = ["file_path", "true_label", "true_label_name", "spoof_probability", "bonafide_probability", "predicted_label", "predicted_label_name"]
    with open(pred_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})
    print(f"Saved predictions to: {pred_csv_path}")
            
    # 4. Metrics
    metrics = calculate_metrics(np.array(probs), np.array(y_true), PREDICTION_THRESHOLD)
    eer, eer_threshold = calculate_eer(y_true, probs)
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold
    
    num_bonafide = sum(1 for y in y_true if y == 0)
    num_spoof = sum(1 for y in y_true if y == 1)
    
    # 5. Save Metrics JSON
    json_path = "outputs/metrics/evaluation_metrics.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    metrics_data = {
        "dataset": "ASVspoof2021_DF_evaluation",
        "model": "AudioDeepfakeCNN",
        "threshold": PREDICTION_THRESHOLD,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "eer": metrics["eer"],
        "eer_threshold": metrics["eer_threshold"],
        "num_samples": len(results),
        "num_bonafide": num_bonafide,
        "num_spoof": num_spoof,
        "evaluation_warning": "PIPELINE VALIDATION RESULTS — NOT FINAL PERFORMANCE. This evaluation used a tiny mock subset of the ASVspoof 2021 DF set."
    }
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    # 6. Plots & Reports
    plot_probability_distribution(y_true, probs)
    plot_roc_curve(y_true, probs)
    plot_confusion_matrix(y_true, probs, PREDICTION_THRESHOLD)
    generate_classification_report(y_true, preds, "outputs/metrics/evaluation_classification_report.txt")
    
    # 7. Per-Attack Analysis
    attack_stats = {}
    for r in results:
        aid = r["attack_id"]
        if aid not in attack_stats:
            attack_stats[aid] = {"y_true": [], "probs": []}
        attack_stats[aid]["y_true"].append(r["true_label"])
        attack_stats[aid]["probs"].append(r["spoof_probability"])
        
    attack_csv = "outputs/metrics/per_attack_results.csv"
    with open(attack_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack_id", "num_samples", "accuracy", "precision", "recall", "f1"])
        for aid, data in attack_stats.items():
            am = calculate_metrics(np.array(data["probs"]), np.array(data["y_true"]), PREDICTION_THRESHOLD)
            writer.writerow([aid, len(data["y_true"]), am["accuracy"], am["precision"], am["recall"], am["f1"]])
            
    # 8. Summary Text
    summary_path = "outputs/metrics/evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("MODEL:\nAudioDeepfakeCNN\n\n")
        f.write("DATASET:\nASVspoof2021 DF evaluation\n\n")
        f.write(f"NUMBER OF SAMPLES:\n{len(results)}\n\n")
        f.write(f"BONAFIDE:\n{num_bonafide}\n\n")
        f.write(f"SPOOF:\n{num_spoof}\n\n")
        f.write(f"THRESHOLD:\n{PREDICTION_THRESHOLD}\n\n")
        f.write(f"ACCURACY:\n{metrics['accuracy']:.4f}\n\n")
        f.write(f"PRECISION:\n{metrics['precision']:.4f}\n\n")
        f.write(f"RECALL:\n{metrics['recall']:.4f}\n\n")
        f.write(f"F1:\n{metrics['f1']:.4f}\n\n")
        f.write(f"ROC-AUC:\n{metrics['roc_auc']:.4f}\n\n")
        f.write(f"EER:\n{metrics['eer']:.4f}\n\n")
        
        f.write("Interpretation\n")
        f.write("==============\n")
        f.write("Accuracy: The proportion of total samples correctly classified.\n")
        f.write("Precision: The proportion of predicted SPOOF samples that were actually SPOOF.\n")
        f.write("Recall: The proportion of actual SPOOF samples correctly identified.\n")
        f.write("F1: The harmonic mean of precision and recall (useful when classes are imbalanced).\n")
        f.write("ROC-AUC: Area under the receiver operating characteristic curve; measures ability to separate classes across all thresholds. Undefined if only one class is present.\n")
        f.write("EER: Equal Error Rate, the point where false positive rate equals false negative rate. Lower is better. Undefined if only one class is present.\n\n")
        f.write("PIPELINE VALIDATION RESULTS — NOT FINAL PERFORMANCE.\n")

    print(f"\nEvaluation Complete! Results saved to outputs/metrics and outputs/plots.")

if __name__ == "__main__":
    main()
