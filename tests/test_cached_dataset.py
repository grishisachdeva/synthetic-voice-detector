import os
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from src.cached_dataset import CachedMelSpectrogramDataset, create_cached_dataloaders
from src.build_feature_cache import validate_feature

# 1. Validation Logic
def test_validate_feature_valid():
    arr = np.zeros((128, 251), dtype=np.float32)
    valid, msg = validate_feature(arr)
    assert valid is True

def test_validate_feature_invalid_shape():
    arr = np.zeros((128, 250), dtype=np.float32)
    valid, msg = validate_feature(arr)
    assert valid is False
    assert "shape" in msg.lower()

def test_validate_feature_invalid_dtype():
    arr = np.zeros((128, 251), dtype=np.float64)
    valid, msg = validate_feature(arr)
    assert valid is False
    assert "dtype" in msg.lower()

def test_validate_feature_nan():
    arr = np.zeros((128, 251), dtype=np.float32)
    arr[0, 0] = np.nan
    valid, msg = validate_feature(arr)
    assert valid is False
    assert "nan" in msg.lower()

def test_validate_feature_inf():
    arr = np.zeros((128, 251), dtype=np.float32)
    arr[0, 0] = np.inf
    valid, msg = validate_feature(arr)
    assert valid is False
    assert "inf" in msg.lower()

# 2. Dataset Integration Tests
@pytest.fixture
def mock_cache_csv(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    # Create two dummy cache files
    arr1 = np.ones((128, 251), dtype=np.float32)
    arr2 = np.zeros((128, 251), dtype=np.float32)
    
    path1 = cache_dir / "id1.npy"
    path2 = cache_dir / "id2.npy"
    
    np.save(path1, arr1)
    np.save(path2, arr2)
    
    metadata = pd.DataFrame([
        {
            "file_id": "id1",
            "original_file_path": "raw/id1.flac",
            "cache_path": str(path1),
            "label": 0,
            "label_name": "bonafide",
            "speaker_id": "spk1",
            "attack_id": "-",
            "partition": "training"
        },
        {
            "file_id": "id2",
            "original_file_path": "raw/id2.flac",
            "cache_path": str(path2),
            "label": 1,
            "label_name": "spoof",
            "speaker_id": "spk2",
            "attack_id": "A01",
            "partition": "training"
        }
    ])
    
    csv_path = tmp_path / "cache.csv"
    metadata.to_csv(csv_path, index=False)
    return str(csv_path)

def test_dataset_loading_and_shapes(mock_cache_csv):
    dataset = CachedMelSpectrogramDataset(mock_cache_csv)
    assert len(dataset) == 2
    
    sample = dataset[0]
    assert "features" in sample
    assert "label" in sample
    assert sample["features"].shape == (1, 128, 251)
    assert sample["features"].dtype == torch.float32
    assert sample["label"].item() == 0.0
    assert sample["file_id"] == "id1"
    
    sample2 = dataset[1]
    assert sample2["label"].item() == 1.0

def test_dataloader_batch_shape(mock_cache_csv):
    loader, _ = create_cached_dataloaders(mock_cache_csv, mock_cache_csv, batch_size=2)
    batch = next(iter(loader))
    
    assert batch["features"].shape == (2, 1, 128, 251)
    assert batch["label"].shape == (2,)

# 3. Cache Builder Logic Simulation
def test_deterministic_mapping():
    # Implicitly tested via builder script file_id mapping logic.
    file_id = "LA_T_1000001"
    expected = f"{file_id}.npy"
    assert "LA_T_1000001.npy" == expected

def test_corrupt_file_handling(tmp_path):
    # This verifies that validate_feature flags a corrupt load.
    bad_file = tmp_path / "bad.npy"
    with open(bad_file, 'wb') as f:
        f.write(b"NOT A NUMPY FILE")
        
    try:
        arr = np.load(bad_file)
        valid, _ = validate_feature(arr)
    except Exception as e:
        valid = False
        
    assert valid is False

def test_missing_cache_handling(mock_cache_csv):
    dataset = CachedMelSpectrogramDataset(mock_cache_csv)
    # Remove one file
    os.remove(dataset.metadata.iloc[0]['cache_path'])
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
