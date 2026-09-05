# utils.py — Reusable utilities for training: seed, metrics, checkpoints, and plotting.

import os
import random
import numpy as np
import torch
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import RANDOM_SEED

def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Configure cuDNN for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calculate classification metrics.
    
    Returns: accuracy, precision, recall, f1, roc_auc.
    Uses NaN if metrics cannot be computed mathematically (e.g. 1 class only).
    """
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    total = len(labels)
    acc = (tp + tn) / total if total > 0 else float('nan')
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float('nan')
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    # Calculate ROC-AUC using sklearn if multiple classes exist
    roc_auc = float('nan')
    if len(np.unique(labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(labels, probs)
        except Exception:
            pass
            
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "model_name": model.__class__.__name__
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint. Returns epoch and best_metric."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", 0.0)

def save_training_history(history: list[dict], path: str):
    """Save history list of dicts to CSV."""
    if not history:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = history[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
