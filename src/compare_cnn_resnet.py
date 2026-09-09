import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_comparison():
    out_dir = "outputs/plots/cnn_vs_resnet_real"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load CNN metrics
    cnn_metrics_path = "outputs/metrics/cnn_real_dev_metrics.json"
    resnet_metrics_path = "outputs/metrics/resnet18_real_dev_metrics.json"
    
    cnn_metrics = json.load(open(cnn_metrics_path))
    resnet_metrics = json.load(open(resnet_metrics_path))
    
    cnn_sel_th = json.load(open("outputs/metrics/cnn_real_selected_threshold.json"))
    resnet_sel_th = json.load(open("outputs/metrics/resnet18_real_selected_threshold.json"))
    
    comp_data = []
    
    for name, m, th in [("AudioDeepfakeCNN", cnn_metrics, cnn_sel_th), ("AudioResNet18", resnet_metrics, resnet_sel_th)]:
        comp_data.append({
            "model": name,
            "parameters": 421825 if name == "AudioDeepfakeCNN" else 11176513,
            "best_epoch": m["best_epoch"],
            "val_loss": -m["best_validation_metric"] if m["best_validation_metric"] < 0 else None,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "F1": m["f1"],
            "ROC_AUC": m["roc_auc"],
            "EER": m["eer"],
            "EER_threshold": m["eer_threshold"],
            "selected_threshold": th["selected_threshold"]
        })
        
    df = pd.DataFrame(comp_data)
    df.to_csv("outputs/metrics/cnn_vs_resnet_real_comparison.csv", index=False)
    
    # Plots
    models = df['model']
    
    metrics_to_plot = [
        ("ROC_AUC", "roc_auc_comparison.png", "ROC-AUC Comparison"),
        ("F1", "f1_comparison.png", "F1 Score Comparison"),
        ("balanced_accuracy", "balanced_accuracy_comparison.png", "Balanced Accuracy Comparison"),
        ("EER", "eer_comparison.png", "EER Comparison (Lower is Better)"),
        ("parameters", "parameter_comparison.png", "Parameter Count Comparison (Log Scale)")
    ]
    
    for col, filename, title in metrics_to_plot:
        plt.figure(figsize=(6, 4))
        bars = plt.bar(models, df[col], color=['blue', 'orange'])
        plt.title(title)
        if col == "parameters":
            plt.yscale('log')
        plt.ylabel(col)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4g}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()
        
    # Statistical Sanity Check (Prediction Agreement)
    cnn_preds_df = pd.read_csv("outputs/predictions/cnn_real_dev_predictions.csv")
    resnet_preds_df = pd.read_csv("outputs/predictions/resnet18_real_dev_predictions.csv")
    
    identical_preds = sum(cnn_preds_df['predicted_label'] == resnet_preds_df['predicted_label'])
    differing_preds = sum(cnn_preds_df['predicted_label'] != resnet_preds_df['predicted_label'])
    
    corr = np.corrcoef(cnn_preds_df['spoof_probability'], resnet_preds_df['spoof_probability'])[0, 1]
    
    agreement_data = {
        "identical_predictions": int(identical_preds),
        "differing_predictions": int(differing_preds),
        "spoof_probability_correlation": float(corr),
        "total_samples": len(cnn_preds_df)
    }
    
    with open("outputs/metrics/cnn_vs_resnet_prediction_agreement.json", "w") as f:
        json.dump(agreement_data, f, indent=4)
        
if __name__ == "__main__":
    generate_comparison()
    print("Comparison complete!")
