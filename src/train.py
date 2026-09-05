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
from src.dataset import create_datasets, create_dataloaders, get_class_weights
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
    
    for name, train_key, val_key, ylabel in metrics_to_plot:
        plt.figure()
        plt.plot(epochs, [h[train_key] for h in history], label=f"Train {name.capitalize()}")
        plt.plot(epochs, [h[val_key] for h in history], label=f"Val {name.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Training and Validation {ylabel}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"training_{name}.png"), dpi=150)
        plt.close()
        
    # ROC-AUC is only validation here
    plt.figure()
    plt.plot(epochs, [h["val_roc_auc"] for h in history], label="Val ROC-AUC", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Validation ROC-AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "validation_roc_auc.png"), dpi=150)
    plt.close()

def plot_confusion_matrix(model, loader, device, threshold=0.5, out_dir="outputs/plots"):
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

def train_model(epochs=EPOCHS, resume_path=None, use_class_weight=USE_CLASS_WEIGHT):
    set_seed()
    
    # 1. Datasets & Loaders
    train_ds, dev_ds, eval_ds = create_datasets()
    train_loader, dev_loader, eval_loader = create_dataloaders(train_ds, dev_ds, eval_ds)
    
    # 2. Model & Device
    model = create_model()
    device = next(model.parameters()).device
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
    # 3. Loss
    if use_class_weight:
        weights = get_class_weights(train_ds)
        # Assuming pos_weight is for SPOOF (index 1)
        # get_class_weights returns [w_bonafide, w_spoof]. The scalar positive weight = w_spoof / w_bonafide
        # Actually BCEWithLogitsLoss pos_weight is number of negative / number of positive.
        pos_weight = get_class_weights(train_ds)
        # Using a safer manual calculation matching the requirement:
        bonafide_count = sum(1 for r in train_ds.metadata if r["label_name"].lower() == "bonafide")
        spoof_count = sum(1 for r in train_ds.metadata if r["label_name"].lower() == "spoof")
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
        
    os.makedirs("outputs/metrics", exist_ok=True)
    log_file = open("outputs/metrics/training_log.txt", "a")
    
    patience_counter = 0
    
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
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric, 
                config={"lr": LEARNING_RATE, "batch_size": BATCH_SIZE}, 
                path=CHECKPOINT_PATH
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
        save_training_history(history, "outputs/metrics/training_history.csv")
        
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
    
    # Generate plots
    plot_training_curves(history)
    
    # Reload best model and evaluate
    print("Validating checkpoint reload...")
    best_model = create_model()
    load_checkpoint(CHECKPOINT_PATH, best_model)
    plot_confusion_matrix(best_model, dev_loader, device)
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--class-weight", action="store_true", help="Enable positive class weighting")
    
    args = parser.parse_args()
    
    # Override globals for this run if specified
    import src.config
    src.config.BATCH_SIZE = args.batch_size
    src.config.LEARNING_RATE = args.learning_rate
    
    train_model(
        epochs=args.epochs, 
        resume_path=args.resume, 
        use_class_weight=args.class_weight or USE_CLASS_WEIGHT
    )
