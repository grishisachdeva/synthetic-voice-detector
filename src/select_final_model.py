import json
import os

def select_model():
    cnn_metrics = {
        "ROC-AUC": 1.0000,
        "F1": 0.9997,
        "Balanced Accuracy": 0.9991,
        "EER": 0.0012,
        "checkpoint": "models/cnn_real/best_cnn_real.pth",
        "threshold": 0.5
    }
    
    resnet_metrics = {
        "ROC-AUC": 0.9998,
        "F1": 0.9975,
        "Balanced Accuracy": 0.9897,
        "EER": 0.0059,
        "checkpoint": "models/resnet18_real/best_resnet18_real.pth",
        "threshold": 0.5  # Need to check
    }
    
    # Selection criteria: ROC-AUC > EER > F1
    winner = "CNN"
    
    selection_report = {
        "selected_model": winner,
        "selected_checkpoint": cnn_metrics["checkpoint"],
        "selection_partition": "development",
        "selection_metrics": cnn_metrics,
        "alternative_model": "ResNet18",
        "alternative_model_metrics": resnet_metrics,
        "selection_reason": "CNN achieved superior ROC-AUC (1.0000 vs 0.9998) and lower EER (0.0012 vs 0.0059) on the development partition."
    }
    
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/final_model_selection.json', 'w') as f:
        json.dump(selection_report, f, indent=4)
        
    with open('outputs/metrics/final_model_selection.txt', 'w') as f:
        f.write("FINAL MODEL SELECTION REPORT\n")
        f.write("============================\n")
        f.write(f"Winner: {winner}\n")
        f.write(f"Checkpoint: {cnn_metrics['checkpoint']}\n")
        f.write("\nReason:\n")
        f.write(selection_report["selection_reason"] + "\n")
        
    print("Model selection complete.")

if __name__ == "__main__":
    select_model()
