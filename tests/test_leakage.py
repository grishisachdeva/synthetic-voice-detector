import os
import json
import pytest
import pandas as pd
from pathlib import Path

def test_no_leakage():
    """
    Ensure that ASVspoof2021 DF paths do NOT appear in:
    - training_metadata.csv
    - development_metadata.csv
    - training_cache.csv
    - development_cache.csv
    """
    target_string = "ASVspoof2021_DF"
    
    files_to_check = [
        "data/metadata/training_metadata.csv",
        "data/metadata/development_metadata.csv",
        "data/processed/mel_cache/metadata/training_cache.csv",
        "data/processed/mel_cache/metadata/development_cache.csv"
    ]
    
    for fpath in files_to_check:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
                assert target_string not in content, f"LEAKAGE DETECTED in {fpath}"

def test_speaker_disjoint():
    """
    Ensure no speaker overlap exists between training and development.
    """
    train_path = "data/metadata/training_metadata.csv"
    dev_path = "data/metadata/development_metadata.csv"
    
    if os.path.exists(train_path) and os.path.exists(dev_path):
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
        
        train_speakers = set(train_df['speaker_id'].unique())
        dev_speakers = set(dev_df['speaker_id'].unique())
        
        overlap = train_speakers.intersection(dev_speakers)
        assert len(overlap) == 0, f"Speaker overlap detected: {overlap}"
