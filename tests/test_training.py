# tests/test_training.py — Unit tests for the training pipeline.

import os
import torch
import pytest
import numpy as np
import tempfile
from src.utils import set_seed, calculate_metrics, save_checkpoint, load_checkpoint, save_training_history
from src.train import train_one_epoch, evaluate_one_epoch
from src.model import AudioDeepfakeCNN
from src.config import LEARNING_RATE

def test_seed_reproducibility():
    set_seed(42)
    a = torch.rand(1).item()
    set_seed(42)
    b = torch.rand(1).item()
    assert a == b

def test_metrics_calculation():
    probs = np.array([0.9, 0.1, 0.8, 0.2, 0.4])
    labels = np.array([1, 0, 1, 0, 0])
    # Threshold 0.5 -> preds = [1, 0, 1, 0, 0]
    metrics = calculate_metrics(probs, labels, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] > 0.9

def test_metrics_one_class():
    probs = np.array([0.9, 0.8])
    labels = np.array([1, 1])
    metrics = calculate_metrics(probs, labels)
    assert metrics["accuracy"] == 1.0
    # ROC-AUC is undefined for one class, should return NaN
    assert np.isnan(metrics["roc_auc"])

def test_checkpoint_save_load():
    model = AudioDeepfakeCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pth")
        
        # Save
        save_checkpoint(model, optimizer, None, epoch=5, best_metric=0.99, config={}, path=ckpt_path)
        assert os.path.exists(ckpt_path)
        
        # Load
        new_model = AudioDeepfakeCNN()
        new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        epoch, metric = load_checkpoint(ckpt_path, new_model, new_opt, None)
        
        assert epoch == 5
        assert metric == 0.99

def test_history_saving():
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_path = os.path.join(tmpdir, "hist.csv")
        history = [
            {"epoch": 1, "loss": 0.5},
            {"epoch": 2, "loss": 0.4}
        ]
        save_training_history(history, hist_path)
        assert os.path.exists(hist_path)
        with open(hist_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 3 # Header + 2 rows

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=4):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            "features": torch.randn(1, 128, 251),
            "label": torch.tensor(float(idx % 2))
        }

def test_train_one_epoch():
    ds = DummyDataset(size=4)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    model = AudioDeepfakeCNN()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cpu")
    
    # Store initial weights to check for updates
    init_weight = model.classifier[-1].weight.clone()
    
    loss, metrics = train_one_epoch(model, loader, criterion, optimizer, device)
    
    assert loss > 0
    assert "accuracy" in metrics
    
    # Check optimizer actually updated weights
    final_weight = model.classifier[-1].weight.clone()
    assert not torch.equal(init_weight, final_weight)

def test_evaluate_one_epoch():
    ds = DummyDataset(size=4)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    model = AudioDeepfakeCNN()
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device("cpu")
    
    loss, metrics = evaluate_one_epoch(model, loader, criterion, device)
    
    assert loss > 0
    assert "accuracy" in metrics
