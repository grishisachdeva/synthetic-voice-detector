# train.py — CNN training loop, evaluation, checkpointing, and plotting.

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM, USE_CLASS_WEIGHT, CHECKPOINT_PATH, PREDICTION_THRESHOLD,
    LR_FACTOR, LR_PATIENCE, LR_MIN, BATCH_SIZE
)
from src.model import create_model
from src.utils import set_seed, calculate_metrics, save_checkpoint, load_checkpoint, save_training_history

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one epoch of training."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device).unsqueeze(1) # shape [B, 1] for BCE
        
        optimizer.zero_grad()
        
        logits = model(features)
        loss = criterion(logits, labels)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
        
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_preds), np.array(all_labels), PREDICTION_THRESHOLD)
    
    return epoch_loss, metrics

def evaluate_one_epoch(model, loader, criterion, device):
    """Run one epoch of evaluation (validation)."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * features.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_preds), np.array(all_labels), PREDICTION_THRESHOLD)
    
    return epoch_loss, metrics

def plot_training_curves(history, out_dir="outputs/plots"):
    """Generate training curve plots."""
    os.makedirs(out_dir, exist_ok=True)
    
    epochs = [h["epoch"] for h in history]
    
    metrics_to_plot = [
        ("loss", "train_loss", "val_loss", "Loss"),
        ("accuracy", "train_accuracy", "val_accuracy", "Accuracy"),
        ("precision", "train_precision", "val_precision", "Precision"),
        ("recall", "train_recall", "val_recall", "Recall"),
        ("f1", "train_f1", "val_f1", "F1 Score")
    ]
    
    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], label='Train Loss')
    plt.plot(epochs, [h["val_loss"] for h in history], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
    plt.close()
    
    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_accuracy"] for h in history], label='Train Accuracy')
    plt.plot(epochs, [h["val_accuracy"] for h in history], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_accuracy.png"), dpi=150)
    plt.close()
    
    # 3. Precision Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_precision"] for h in history], label='Train Precision')
    plt.plot(epochs, [h["val_precision"] for h in history], label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_precision.png"), dpi=150)
    plt.close()
    
    # 4. Recall Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_recall"] for h in history], label='Train Recall')
    plt.plot(epochs, [h["val_recall"] for h in history], label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_recall.png"), dpi=150)
    plt.close()
    
    # 5. F1 Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_f1"] for h in history], label='Train F1')
    plt.plot(epochs, [h["val_f1"] for h in history], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_f1.png"), dpi=150)
    plt.close()
    
    # 6. ROC-AUC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["val_roc_auc"] for h in history], label='Validation ROC-AUC', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title('Validation ROC-AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "validation_roc_auc.png"), dpi=150)
    plt.close()

def plot_confusion_matrix(model, loader, device, threshold=0.5, out_dir="outputs/plots/cnn"):
    """Generate and save confusion matrix on the validation set."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            if probs.ndim == 0:
                probs = np.expand_dims(probs, 0)
            preds = (probs >= threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))
            
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BONAFIDE", "SPOOF"])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues")
        plt.title("Validation Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "validation_confusion_matrix.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Failed to generate confusion matrix: {e}")

def train_model(model_name="cnn", run_name=None, epochs=EPOCHS, resume_path=None, use_class_weight=USE_CLASS_WEIGHT):
    set_seed()
    
    if run_name is None:
        run_name = model_name
        
    # Configure paths based on run_name
    ckpt_dir = f"models/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"best_{run_name}.pth")
    
    os.makedirs("outputs/metrics", exist_ok=True)
    hist_path = f"outputs/metrics/{run_name}_training_history.csv"
    log_path = f"outputs/metrics/{run_name}_training_log.txt"
    metrics_path = f"outputs/metrics/{run_name}_final_training_metrics.json"
    
    plots_dir = f"outputs/plots/{run_name}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Datasets & Loaders
    # 1. Datasets & Loaders
    from src.cached_dataset import create_cached_dataloaders
    train_csv = "data/processed/mel_cache/metadata/training_cache.csv"
    dev_csv = "data/processed/mel_cache/metadata/development_cache.csv"
    
    # We will pass max_files via global state or argument if needed, but let's just handle it.
    import pandas as pd
    from pathlib import Path
    
    # Check if max_files is configured (via argparse in __main__)
    import src.config
    max_files = getattr(src.config, 'MAX_FILES', None)
    
    if max_files:
        train_df = pd.read_csv(train_csv).head(max_files)
        train_csv_trunc = "data/processed/mel_cache/metadata/training_cache_truncated.csv"
        train_df.to_csv(train_csv_trunc, index=False)
        train_csv = train_csv_trunc
        
        dev_df = pd.read_csv(dev_csv).head(max_files)
        dev_csv_trunc = "data/processed/mel_cache/metadata/development_cache_truncated.csv"
        dev_df.to_csv(dev_csv_trunc, index=False)
        dev_csv = dev_csv_trunc

    train_loader, dev_loader = create_cached_dataloaders(str(train_csv), str(dev_csv), batch_size=BATCH_SIZE)
    
    # 2. Model & Device
    model = create_model(model_name=model_name)
    device = next(model.parameters()).device
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
    print(f"Dataset: ASVspoof 2019 LA (Cached)")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Development samples: {len(dev_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Optimizer: AdamW")
    print(f"Scheduler: ReduceLROnPlateau")
    print(f"Class weighting: {use_class_weight}")
  
    # 3. Loss
    if use_class_weight:
        # Calculate weights dynamically from cached dataset DataFrame
        df = train_loader.dataset.df
        bonafide_count = len(df[df["label"] == 0])
        spoof_count = len(df[df["label"] == 1])
        scalar_weight = (bonafide_count / spoof_count) if spoof_count > 0 else 1.0
        pos_weight = torch.tensor([scalar_weight], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss Configuration: BCEWithLogitsLoss(pos_weight={scalar_weight:.4f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss Configuration: BCEWithLogitsLoss (Unweighted)")
        
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 5. Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN
    )
    
    start_epoch = 0
    best_metric = -float('inf') # We maximize ROC-AUC or F1
    best_criterion_name = ""
    history = []

    
    if resume_path:
        print(f"Resuming from {resume_path}...")
        start_epoch, best_metric = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"Resumed at epoch {start_epoch} with best metric {best_metric:.4f}")
        
    log_file = open(log_path, "a")
    
    patience_counter = 0
    best_epoch = 0
    
    # 6. Training Loop
    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        log_file.write(f"\nEpoch {epoch}/{epochs}\n")
        
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_one_epoch(model, dev_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Determine best metric
        if not np.isnan(val_metrics["roc_auc"]):
            current_metric = val_metrics["roc_auc"]
            metric_name = "ROC-AUC"
        elif not np.isnan(val_metrics["f1"]):
            current_metric = val_metrics["f1"]
            metric_name = "F1"
        else:
            current_metric = -val_loss # Maximize negative loss
            metric_name = "Negative Loss"
            
        best_criterion_name = metric_name
        
        updated_checkpoint = False
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric, 
                config={"lr": LEARNING_RATE, "batch_size": BATCH_SIZE}, 
                path=ckpt_path
            )
            updated_checkpoint = True
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Logging
        hist_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "learning_rate": current_lr
        }
        history.append(hist_entry)
        save_training_history(history, hist_path)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc:  {train_metrics['accuracy']:.4f} | Val Acc:  {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"LR: {current_lr:.2e} | Best check: {updated_checkpoint} | Early stop count: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        log_file.write(f"Train loss: {train_loss:.4f}\nValidation loss: {val_loss:.4f}\n")
        log_file.write(f"Train accuracy: {train_metrics['accuracy']:.4f}\nValidation accuracy: {val_metrics['accuracy']:.4f}\n")
        log_file.write(f"Precision: {val_metrics['precision']:.4f}\nRecall: {val_metrics['recall']:.4f}\nF1: {val_metrics['f1']:.4f}\nROC-AUC: {val_metrics['roc_auc']:.4f}\n")
        log_file.write(f"Learning rate: {current_lr:.2e}\n")
        log_file.write(f"Best checkpoint updated: {updated_checkpoint}\n")
        log_file.write(f"Early stopping counter increased: {patience_counter > 0} ({patience_counter})\n")
        log_file.flush()
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            log_file.write(f"Early stopping triggered at epoch {epoch}\n")
            break
            
    print(f"\nTraining completed. Best model selected using Validation {best_criterion_name}.")
    log_file.write(f"\nTraining completed. Best model selected using Validation {best_criterion_name}.\n")
    log_file.close()
    
    # Save final JSON metrics
    import json
    if history:
        best_hist = next((h for h in history if h["epoch"] == best_epoch), history[-1])
        final_hist = history[-1]
        
        final_metrics = {
            "model_name": model_name,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_metric": best_metric,
            "best_validation_roc_auc": best_hist.get("val_roc_auc", float('nan')),
            "best_validation_f1": best_hist.get("val_f1", float('nan')),
            "final_train_loss": final_hist.get("train_loss", float('nan')),
            "final_val_loss": final_hist.get("val_loss", float('nan')),
            "final_train_accuracy": final_hist.get("train_accuracy", float('nan')),
            "final_val_accuracy": final_hist.get("val_accuracy", float('nan')),
            "final_train_f1": final_hist.get("train_f1", float('nan')),
            "final_val_f1": final_hist.get("val_f1", float('nan')),
            "warning": "The current dataset is a mock subset and these values are not representative of final model performance."
        }
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=4)
    
    # Generate plots
    plot_training_curves(history, out_dir=plots_dir)
    
    # Reload best model and evaluate
    print("Validating checkpoint reload...")
    best_model = create_model(model_name=model_name)
    load_checkpoint(ckpt_path, best_model)
    plot_confusion_matrix(best_model, dev_loader, device, out_dir=plots_dir)
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Models")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet18"], help="Model architecture to train")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the run (determines output folders)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--class-weight", action="store_true", help="Enable positive class weighting")
    parser.add_argument("--max-files", type=int, default=None, help="Truncate dataset for smoke tests")
    
    args = parser.parse_args()
    
    # Override globals for this run if specified
    import src.config
    src.config.BATCH_SIZE = args.batch_size
    src.config.MAX_FILES = args.max_files
    if args.learning_rate is not None:
        src.config.LEARNING_RATE = args.learning_rate
    elif args.model == "resnet18":
        src.config.LEARNING_RATE = src.config.RESNET_LEARNING_RATE
        
    train_model(
        model_name=args.model,
        run_name=args.run_name,
        epochs=args.epochs, 
        resume_path=args.resume, 
        use_class_weight=args.class_weight or USE_CLASS_WEIGHT
    )
