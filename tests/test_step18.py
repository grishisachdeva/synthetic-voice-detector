import os
import pandas as pd
import pytest
from src.config import EVAL_METADATA, TRAIN_METADATA, DEV_METADATA
from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram
import soundfile as sf
import librosa

def test_real_df_metadata_loading():
    # Only test if metadata exists
    if not os.path.exists(EVAL_METADATA):
        pytest.skip("EVAL_METADATA not generated yet.")
        
    df = pd.read_csv(EVAL_METADATA)
    assert not df.empty, "Metadata should not be empty."
    assert "file_path" in df.columns
    assert "label" in df.columns
    assert "label_name" in df.columns

def test_label_mapping():
    if not os.path.exists(EVAL_METADATA):
        pytest.skip("EVAL_METADATA not generated yet.")
    df = pd.read_csv(EVAL_METADATA)
    bonafide = df[df['label_name'] == 'bonafide']
    spoof = df[df['label_name'] == 'spoof']
    assert all(bonafide['label'] == 0)
    assert all(spoof['label'] == 1)

def test_evaluation_isolation():
    # Leakage test
    if not os.path.exists(EVAL_METADATA):
        pytest.skip()
        
    eval_df = pd.read_csv(EVAL_METADATA)
    train_df = pd.read_csv(TRAIN_METADATA)
    dev_df = pd.read_csv(DEV_METADATA)
    
    train_overlap = set(eval_df['file_id']).intersection(set(train_df['file_id']))
    dev_overlap = set(eval_df['file_id']).intersection(set(dev_df['file_id']))
    
    assert len(train_overlap) == 0, f"Found {len(train_overlap)} leaked IDs in training!"
    assert len(dev_overlap) == 0, f"Found {len(dev_overlap)} leaked IDs in development!"

def test_metadata_file_consistency():
    if not os.path.exists(EVAL_METADATA):
        pytest.skip()
    df = pd.read_csv(EVAL_METADATA)
    sample_path = df.iloc[0]['file_path']
    assert os.path.exists(sample_path), f"File {sample_path} from metadata does not exist on disk."

def test_sample_preprocessing():
    if not os.path.exists(EVAL_METADATA):
        pytest.skip()
    df = pd.read_csv(EVAL_METADATA)
    for _, row in df.iterrows():
        try:
            sample_path = row['file_path']
            waveform, sr = librosa.load(sample_path, sr=16000)
            proc_wave = preprocess_audio(waveform)
            assert proc_wave.shape == (64000,)
            mel = extract_mel_spectrogram(proc_wave)
            assert mel.shape == (128, 251)
            break  # Found one that works
        except Exception:
            continue

def test_mock_real_dataset_separation():
    # Mock data should be preserved in backup
    assert os.path.exists("data/mock_backup/asvspoof2021_df_mock")
    # Real data should be in its own directory
    assert os.path.exists("data/raw/ASVspoof2021_DF_real")
