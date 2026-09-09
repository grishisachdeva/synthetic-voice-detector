import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.model import AudioDeepfakeCNN
from src.cached_dataset import create_cached_dataloaders
from src.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_NORM, RANDOM_SEED
)

THRESHOLD = 0.5

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(float)
    acc = accuracy_score(y_true, y_pred)
    
    # Handle single class edge-cases during very early batches
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_probs)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        roc_auc = float('nan')
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return acc, precision, recall, f1, roc_auc

def train_epoch(model, dataloader, criterion, optimizer, device, clip_norm, use_spec_augment=False):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    # For SpecAugment (time and freq masking logic)
    # We will implement a simplified SpecAugment if requested.
    
    for batch in dataloader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        # Primitive SpecAugment applied dynamically on features if enabled
        if use_spec_augment:
            # Masking features. features shape: (B, 1, 128, 251)
            b, c, f, t = features.shape
            # Frequency masking
            f_mask_max = int(f * 0.15)
            # Time masking
            t_mask_max = int(t * 0.15)
            for i in range(b):
                f0 = np.random.randint(0, f - f_mask_max)
                f_mask = np.random.randint(1, f_mask_max + 1)
                t0 = np.random.randint(0, t - t_mask_max)
                t_mask = np.random.randint(1, t_mask_max + 1)
                features[i, 0, f0:f0+f_mask, :] = 0.0
                features[i, 0, :, t0:t0+t_mask] = 0.0

        optimizer.zero_grad()
        
        # Model returns (batch, 1) logits
        logits = model(features).squeeze(1)
        loss = criterion(logits, labels)
        
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    acc, prec, rec, f1, auc = compute_metrics(np.array(all_targets), np.array(all_probs), THRESHOLD)
    return epoch_loss, acc, prec, rec, f1, auc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(features).squeeze(1)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * features.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    acc, prec, rec, f1, auc = compute_metrics(np.array(all_targets), np.array(all_probs), THRESHOLD)
    return epoch_loss, acc, prec, rec, f1, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-loss', action='store_true', help="Use pos_weight class balancing")
    parser.add_argument('--spec-augment', action='store_true', help="Use SpecAugment data augmentation")
    parser.add_argument('--max-files', type=int, default=None, help="Truncate dataset for dry runs")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Paths
    train_csv = Path("data/processed/mel_cache/metadata/training_cache.csv")
    dev_csv = Path("data/processed/mel_cache/metadata/development_cache.csv")
    
    # Calculate pos_weight dynamically
    train_df = pd.read_csv(train_csv)
    if args.max_files:
        train_df = train_df.head(args.max_files)
        # We need to save the truncated csv temporarily to pass to the dataloader
        train_csv = "data/processed/mel_cache/metadata/training_cache_truncated.csv"
        train_df.to_csv(train_csv, index=False)
        
        dev_df = pd.read_csv(dev_csv).head(args.max_files)
        dev_csv = "data/processed/mel_cache/metadata/development_cache_truncated.csv"
        dev_df.to_csv(dev_csv, index=False)
    
    # SPOOF = 1 (positive), BONAFIDE = 0 (negative)
    spoof_count = len(train_df[train_df['label'] == 1])
    bonafide_count = len(train_df[train_df['label'] == 0])
    print(f"Training distribution: SPOOF={spoof_count}, BONAFIDE={bonafide_count}")
    
    pos_weight = None
    if args.weight_loss:
        # positive class = SPOOF
        pos_weight_val = bonafide_count / max(spoof_count, 1)
        print(f"Calculated pos_weight (BONAFIDE / SPOOF): {pos_weight_val:.4f}")
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    else:
        print("Using Unweighted BCEWithLogitsLoss")

    train_loader, dev_loader = create_cached_dataloaders(str(train_csv), str(dev_csv), batch_size=BATCH_SIZE)
    
    model = AudioDeepfakeCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("models/cnn_real", exist_ok=True)
    os.makedirs("models/cnn_real_specaug", exist_ok=True)
    
    # Suffix for files
    suffix = "_specaug" if args.spec_augment else ""
    log_file = f"outputs/metrics/cnn_real{suffix}_training_log.txt"
    history_file = f"outputs/metrics/cnn_real{suffix}_training_history.csv"
    model_path = f"models/cnn_real{suffix}/best_cnn_real.pth"

    best_val_roc_auc = -1.0
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    best_epoch = -1
    early_stop_counter = 0

    history = []
    
    with open(log_file, "w") as f:
        f.write("Epoch,TrainLoss,ValLoss,ValAcc,ValPrec,ValRec,ValF1,ValROC,LR,CheckStatus,ESCounter\n")

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        t_loss, t_acc, t_prec, t_rec, t_f1, t_auc = train_epoch(model, train_loader, criterion, optimizer, device, GRADIENT_CLIP_NORM, use_spec_augment=args.spec_augment)
        v_loss, v_acc, v_prec, v_rec, v_f1, v_auc = validate_epoch(model, dev_loader, criterion, device)
        
        lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_loss)
        
        is_best = False
        # Selection logic: ROC-AUC > F1 > Loss
        if not np.isnan(v_auc) and v_auc > best_val_roc_auc:
            is_best = True
            best_val_roc_auc = v_auc
            best_val_f1 = v_f1
            best_val_loss = v_loss
        elif np.isnan(v_auc) and not np.isnan(v_f1) and v_f1 > best_val_f1:
            is_best = True
            best_val_f1 = v_f1
            best_val_loss = v_loss
        elif np.isnan(v_auc) and np.isnan(v_f1) and v_loss < best_val_loss:
            is_best = True
            best_val_loss = v_loss

        if is_best:
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_val_roc_auc if not np.isnan(best_val_roc_auc) else (best_val_f1 if not np.isnan(best_val_f1) else best_val_loss),
                'config': {'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'spec_augment': args.spec_augment, 'weighted': args.weight_loss},
                'model_name': 'AudioDeepfakeCNN',
                'dataset_name': 'ASVspoof2019_LA_Cached'
            }, model_path)
            chk_status = "SAVED"
        else:
            early_stop_counter += 1
            chk_status = "-"
            
        ep_time = time.time() - ep_start
        print(f"Epoch {epoch:02d}/{EPOCHS} [{ep_time:.1f}s] T-Loss: {t_loss:.4f} V-Loss: {v_loss:.4f} V-AUC: {v_auc:.4f} V-F1: {v_f1:.4f} ES: {early_stop_counter}/{EARLY_STOPPING_PATIENCE} {chk_status}")
        
        history.append({
            'epoch': epoch,
            'train_loss': t_loss, 'val_loss': v_loss,
            'train_accuracy': t_acc, 'val_accuracy': v_acc,
            'train_precision': t_prec, 'val_precision': v_prec,
            'train_recall': t_rec, 'val_recall': v_rec,
            'train_f1': t_f1, 'val_f1': v_f1,
            'val_roc_auc': v_auc, 'learning_rate': lr
        })
        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{t_loss:.6f},{v_loss:.6f},{v_acc:.4f},{v_prec:.4f},{v_rec:.4f},{v_f1:.4f},{v_auc:.4f},{lr:.2e},{chk_status},{early_stop_counter}\n")

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")
    print(f"Best Epoch: {best_epoch} | ROC-AUC: {best_val_roc_auc:.4f} | F1: {best_val_f1:.4f}")
    
    pd.DataFrame(history).to_csv(history_file, index=False)

if __name__ == "__main__":
    main()
