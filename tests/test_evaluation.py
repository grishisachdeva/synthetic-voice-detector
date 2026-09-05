# tests/test_evaluation.py — Unit tests for the evaluation pipeline.

import os
import torch
import pytest
import numpy as np
import tempfile
import json
from src.evaluate import (
    analyze_thresholds, 
    calculate_eer, 
    plot_probability_distribution, 
    plot_roc_curve, 
    plot_confusion_matrix,
    generate_classification_report
)

def test_analyze_thresholds():
    y_true = [1, 0, 1, 0]
    probs = [0.9, 0.1, 0.4, 0.6]
    thresholds = [0.3, 0.5, 0.8]
    
    results = analyze_thresholds(y_true, probs, thresholds)
    assert len(results) == 3
    
    # At thresh 0.5:
    # Preds: [1, 0, 0, 1]
    # TP: 1 (idx 0)
    # TN: 1 (idx 1)
    # FP: 1 (idx 3)
    # FN: 1 (idx 2)
    # precision = 1/2 = 0.5
    res_05 = [r for r in results if r["threshold"] == 0.5][0]
    assert res_05["precision"] == 0.5
    assert res_05["recall"] == 0.5
    assert res_05["fpr"] == 0.5
    assert res_05["fnr"] == 0.5

def test_calculate_eer():
    # Perfect separation
    y_true = [1, 1, 0, 0]
    probs = [0.9, 0.8, 0.2, 0.1]
    eer, eer_thresh = calculate_eer(y_true, probs)
    assert eer == 0.0
    assert 0.2 <= eer_thresh <= 0.8

def test_calculate_eer_one_class():
    y_true = [1, 1, 1]
    probs = [0.9, 0.8, 0.7]
    eer, eer_thresh = calculate_eer(y_true, probs)
    assert np.isnan(eer)
    assert np.isnan(eer_thresh)

def test_plots_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        y_true = [1, 1, 0, 0]
        probs = [0.9, 0.8, 0.2, 0.1]
        
        plot_probability_distribution(y_true, probs, out_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "probability_distribution.png"))
        
        plot_roc_curve(y_true, probs, out_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "test_roc_curve.png"))
        
        plot_confusion_matrix(y_true, probs, out_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "test_confusion_matrix.png"))
        
def test_classification_report_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        y_true = [1, 1, 0, 0]
        preds = [1, 0, 0, 1]
        out_path = os.path.join(tmpdir, "report.txt")
        generate_classification_report(y_true, preds, out_path)
        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            content = f.read()
            assert "Evaluation Classification Report" in content
            assert "BONAFIDE" in content
            assert "SPOOF" in content
