import os
import torch
import numpy as np
from pathlib import Path
from src.train_cnn_real import compute_metrics
from src.evaluate_cnn_real import compute_eer

def test_compute_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.9, 0.8, 0.4, 0.3])
    # threshold 0.5: preds = [0, 1, 1, 0, 0]
    
    acc, prec, rec, f1, auc = compute_metrics(y_true, y_probs)
    
    # 0 = 0 (TN), 1 = 1 (TP), 1 = 1 (TP), 0 = 0 (TN), 1 = 0 (FN)
    # TP=2, TN=2, FP=0, FN=1 -> Acc=4/5=0.8
    # Prec = 2 / 2 = 1.0
    # Rec = 2 / 3 = 0.666
    
    assert acc == 0.8
    assert prec == 1.0
    assert abs(rec - 2/3) < 1e-5
    
def test_eer_computation():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.6, 0.9])
    
    eer, eer_thresh = compute_eer(y_true, y_scores)
    # perfectly separable at 0.5
    assert eer == 0.0
    assert 0.4 <= eer_thresh <= 0.6
    
def test_pos_weight_logic():
    bonafide_count = 2580
    spoof_count = 22800
    pos_weight = bonafide_count / spoof_count
    
    assert abs(pos_weight - 0.11315) < 1e-4

def test_model_cpu_compatibility():
    from src.model import AudioDeepfakeCNN
    model = AudioDeepfakeCNN()
    x = torch.randn(2, 1, 128, 251)
    out = model(x)
    assert out.shape == (2, 1)
