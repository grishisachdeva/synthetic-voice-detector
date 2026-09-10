import pytest
import os
import json
import pandas as pd
import torch

def test_frozen_integrity():
    # Test that original hashes match
    assert os.path.exists("outputs/metrics/initial_integrity_hashes.json")
    with open("outputs/metrics/initial_integrity_hashes.json", "r") as f:
        initial = json.load(f)
        
    import hashlib
    def md5(p):
        h = hashlib.md5()
        with open(p, "rb") as fl:
            for c in iter(lambda: fl.read(4096), b""): h.update(c)
        return h.hexdigest()
        
    assert md5("models/cnn_real/best_cnn_real.pth") == initial["checkpoint"]
    assert md5("outputs/metrics/final_frozen_configuration.json") == initial["frozen_configuration"]
    assert md5("outputs/metrics/final_results.json") == initial["final_results"]

def test_robustness_metrics_exist():
    assert os.path.exists("outputs/metrics/final_robustness_results.csv")
    df = pd.read_csv("outputs/metrics/final_robustness_results.csv")
    assert 'condition' in df.columns
    assert 'f1' in df.columns
    
def test_manifest_fairness():
    assert os.path.exists("outputs/metrics/robustness_sample_manifest.csv")
    df = pd.read_csv("outputs/metrics/robustness_sample_manifest.csv")
    assert len(df['condition'].unique()) >= 10
    # Check that base samples are same across conditions
    clean_ids = set(df[df['condition'] == 'clean']['source_file_id'])
    for cond in df['condition'].unique():
        if cond != 'compression':
            cond_ids = set(df[df['condition'] == cond]['source_file_id'])
            assert clean_ids == cond_ids, f"Condition {cond} does not use the identical base sample subset!"

def test_error_analysis_files():
    assert os.path.exists("outputs/metrics/final_error_statistics.json")
    assert os.path.exists("outputs/metrics/final_attack_error_analysis.csv")
    assert os.path.exists("outputs/metrics/df_exclusion_summary.txt")
    assert os.path.exists("outputs/metrics/final_confidence_by_outcome.csv")

def test_threshold_was_not_tuned():
    with open("outputs/metrics/final_frozen_configuration.json", "r") as f:
        config = json.load(f)
    assert config["classification_threshold"] == 0.5
