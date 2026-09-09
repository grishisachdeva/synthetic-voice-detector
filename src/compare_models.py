# compare_models.py — Controlled architecture comparison script.

import os
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("         MODEL COMPARISON")
    print("=" * 60)

    cnn_metrics_path = "outputs/metrics/cnn_final_training_metrics.json"
    resnet_metrics_path = "outputs/metrics/resnet18_final_training_metrics.json"
    
    # We also hardcode parameters as per task 15
    cnn_params = 421825
    resnet_params = 11170753
    
    cnn_data = None
    resnet_data = None
    
    if os.path.exists(cnn_metrics_path):
        with open(cnn_metrics_path, "r") as f:
            cnn_data = json.load(f)
            
    if os.path.exists(resnet_metrics_path):
        with open(resnet_metrics_path, "r") as f:
            resnet_data = json.load(f)
            
    # Prepare comparison CSV
    csv_path = "outputs/metrics/model_comparison.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    headers = [
        "model", "parameters", "epochs", "best_epoch", "best_val_loss", 
        "best_val_accuracy", "best_val_precision", "best_val_recall", 
        "best_val_f1", "best_val_roc_auc", "training_time_seconds", "status"
    ]
    
    rows = []
    
    def process_model(name, data, params):
        if not data:
            return {
                "model": name, "parameters": params, "epochs": "", "best_epoch": "", 
                "best_val_loss": "", "best_val_accuracy": "", "best_val_precision": "", 
                "best_val_recall": "", "best_val_f1": "", "best_val_roc_auc": "", 
                "training_time_seconds": "", "status": "UNAVAILABLE"
            }
            
        epochs = data.get("epochs_completed", 0)
        status = "SMOKE_TEST" if epochs <= 2 else "FULL_TRAINING"
        
        return {
            "model": name,
            "parameters": params,
            "epochs": epochs,
            "best_epoch": data.get("best_epoch", ""),
            "best_val_loss": data.get("final_val_loss", ""),
            "best_val_accuracy": data.get("final_val_accuracy", ""),
            "best_val_precision": data.get("best_validation_precision", ""), # Note: we didn't save precision best explicitly but we can leave blank if not found
            "best_val_recall": data.get("best_validation_recall", ""),
            "best_val_f1": data.get("best_validation_f1", ""),
            "best_val_roc_auc": data.get("best_validation_roc_auc", ""),
            "training_time_seconds": "", # Not rigorously tracked on CPU
            "status": status
        }
        
    cnn_row = process_model("CNN", cnn_data, cnn_params)
    resnet_row = process_model("ResNet18", resnet_data, resnet_params)
    rows.extend([cnn_row, resnet_row])
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Comparison CSV saved to {csv_path}")
    
    # Ensure plots dir exists
    os.makedirs("outputs/plots", exist_ok=True)
    
    def plot_bar(metric_key, title, filename):
        models = ["CNN", "ResNet18"]
        values = []
        for d in [cnn_data, resnet_data]:
            if d and metric_key in d:
                values.append(d[metric_key])
            else:
                values.append(0.0)
                
        plt.figure(figsize=(6, 5))
        bars = plt.bar(models, values, color=['skyblue', 'lightgreen'])
        plt.title(f"{title}\n(Pipeline validation only — not final performance)")
        plt.ylabel(title)
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')
                
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f"outputs/plots/{filename}", dpi=150)
        plt.close()
        
    plot_bar("final_val_accuracy", "Best Validation Accuracy", "model_comparison_accuracy.png")
    plot_bar("best_validation_f1", "Best Validation F1", "model_comparison_f1.png")
    plot_bar("best_validation_roc_auc", "Best Validation ROC-AUC", "model_comparison_roc_auc.png")
    
    # Parameters plot
    plt.figure(figsize=(6, 5))
    bars = plt.bar(["CNN", "ResNet18"], [cnn_params, resnet_params], color=['lightcoral', 'gold'])
    plt.title("Model Parameter Count Comparison")
    plt.ylabel("Number of Parameters")
    plt.yscale('log') # Log scale is better since 11M vs 400K
    
    for bar, param in zip(bars, [cnn_params, resnet_params]):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{param:,}', ha='center', va='bottom')
        
    plt.savefig("outputs/plots/model_parameter_comparison.png", dpi=150)
    plt.close()
    
    print("Comparison plots generated in outputs/plots/")
    
    print("\n--- Overfitting Check ---")
    def check_overfit(name, data):
        if not data:
            return "Unavailable"
        t_loss = data.get("final_train_loss", float('nan'))
        v_loss = data.get("final_val_loss", float('nan'))
        if t_loss < v_loss - 0.1: # simple heuristic
            return f"Overfitting observed (Train: {t_loss:.4f} < Val: {v_loss:.4f})"
        else:
            return f"No severe overfitting (Train: {t_loss:.4f} vs Val: {v_loss:.4f})"
            
    print(f"CNN: {check_overfit('CNN', cnn_data)}")
    print(f"ResNet18: {check_overfit('ResNet18', resnet_data)}")
    print("\nNote: Overfitting on 6-file mock data is mathematically expected and not a meaningful research result.")

if __name__ == "__main__":
    main()
