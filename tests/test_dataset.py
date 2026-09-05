# tests/test_dataset.py — Unit tests for PyTorch Dataset and DataLoader pipeline.

import os
import csv
import numpy as np
import torch
import pytest
import soundfile as sf

from src.config import SAMPLE_RATE, N_MELS, HOP_LENGTH, TARGET_SAMPLES
from src.dataset import (
    AudioDeepfakeDataset,
    _validate_label,
    get_class_distribution,
    get_class_weights,
    check_speaker_overlap,
    augment_waveform,
    create_dataloaders,
)

EXPECTED_TIME_FRAMES = 1 + (TARGET_SAMPLES // HOP_LENGTH)  # 251


def _make_meta(tmp_path, n=3, label="bonafide", speaker="SPK01", attack="-"):
    """Create n tiny FLAC files and return a metadata list matching the real CSV schema."""
    meta = []
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    for i in range(n):
        fpath = str(tmp_path / f"test_{label}_{i}.flac")
        sf.write(fpath, audio, sr, format="FLAC", subtype="PCM_16")
        meta.append({
            "file_path": fpath,
            "file_id": f"TEST_{i}",
            "speaker_id": speaker,
            "label": str(0 if label == "bonafide" else 1),
            "label_name": label,
            "attack_id": attack,
            "dataset_partition": "train",
        })
    return meta


# ── 1. Dataset initialization ────────────────────────────────
def test_dataset_init(tmp_path):
    meta = _make_meta(tmp_path)
    ds = AudioDeepfakeDataset(meta, split="train", training=False)
    assert ds is not None


# ── 2. Dataset length ────────────────────────────────────────
def test_dataset_length(tmp_path):
    meta = _make_meta(tmp_path, n=5)
    ds = AudioDeepfakeDataset(meta, split="train", training=False)
    assert len(ds) == 5


# ── 3. Valid label mapping ───────────────────────────────────
def test_valid_labels():
    assert _validate_label("bonafide", "test.flac") == 0
    assert _validate_label("spoof", "test.flac") == 1
    assert _validate_label("BONAFIDE", "test.flac") == 0
    assert _validate_label("SPOOF", "test.flac") == 1


# ── 4. Invalid label handling ────────────────────────────────
def test_invalid_label():
    with pytest.raises(ValueError, match="Invalid label"):
        _validate_label("unknown", "test.flac")


# ── 5. Returned dictionary keys ─────────────────────────────
def test_returned_keys(tmp_path):
    meta = _make_meta(tmp_path, n=1)
    ds = AudioDeepfakeDataset(meta, split="train", training=False)
    sample = ds[0]
    assert "features" in sample
    assert "label" in sample
    assert "path" in sample
    assert "speaker_id" in sample
    assert "attack_id" in sample


# ── 6. Feature tensor dtype ──────────────────────────────────
def test_feature_dtype(tmp_path):
    meta = _make_meta(tmp_path, n=1)
    ds = AudioDeepfakeDataset(meta, split="dev", training=False)
    sample = ds[0]
    assert sample["features"].dtype == torch.float32


# ── 7. Feature tensor shape ──────────────────────────────────
def test_feature_shape(tmp_path):
    meta = _make_meta(tmp_path, n=1)
    ds = AudioDeepfakeDataset(meta, split="dev", training=False)
    sample = ds[0]
    assert sample["features"].shape == torch.Size([1, N_MELS, EXPECTED_TIME_FRAMES])


# ── 8. Label dtype ───────────────────────────────────────────
def test_label_dtype(tmp_path):
    meta = _make_meta(tmp_path, n=1)
    ds = AudioDeepfakeDataset(meta, split="dev", training=False)
    sample = ds[0]
    assert sample["label"].dtype == torch.float32


# ── 9. Training augmentation behavior ────────────────────────
def test_augmentation_runs(tmp_path):
    """Augmented waveform should still produce a valid spectrogram."""
    meta = _make_meta(tmp_path, n=1)
    ds = AudioDeepfakeDataset(meta, split="train", training=True)
    sample = ds[0]
    assert sample["features"].shape == torch.Size([1, N_MELS, EXPECTED_TIME_FRAMES])
    assert not torch.isnan(sample["features"]).any()


# ── 10. Validation determinism ────────────────────────────────
def test_validation_determinism(tmp_path):
    meta = _make_meta(tmp_path, n=1)
    ds = AudioDeepfakeDataset(meta, split="dev", training=False)
    s1 = ds[0]
    s2 = ds[0]
    assert torch.allclose(s1["features"], s2["features"], atol=1e-6)


# ── 11. DataLoader batch shape ────────────────────────────────
def test_dataloader_batch_shape(tmp_path):
    meta = _make_meta(tmp_path, n=4)
    ds = AudioDeepfakeDataset(meta, split="train", training=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["features"].shape == torch.Size([2, 1, N_MELS, EXPECTED_TIME_FRAMES])
    assert batch["label"].shape == torch.Size([2])


# ── 12. Empty metadata handling ──────────────────────────────
def test_empty_metadata_raises():
    with pytest.raises(ValueError, match="empty"):
        AudioDeepfakeDataset([], split="train", training=False)


# ── 13. Missing audio handling ────────────────────────────────
def test_missing_audio_raises(tmp_path):
    meta = [{"file_path": str(tmp_path / "nonexistent.flac"), "label_name": "bonafide",
             "speaker_id": "", "attack_id": ""}]
    ds = AudioDeepfakeDataset(meta, split="train", training=False)
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


# ── 14. Speaker overlap detection ─────────────────────────────
def test_speaker_overlap_detection():
    meta_a = [{"speaker_id": "S1"}, {"speaker_id": "S2"}]
    meta_b = [{"speaker_id": "S2"}, {"speaker_id": "S3"}]
    overlap = check_speaker_overlap(meta_a, meta_b, "A", "B")
    assert overlap == {"S2"}


def test_no_speaker_overlap():
    meta_a = [{"speaker_id": "S1"}]
    meta_b = [{"speaker_id": "S2"}]
    overlap = check_speaker_overlap(meta_a, meta_b, "A", "B")
    assert len(overlap) == 0


# ── 15. Class distribution ───────────────────────────────────
def test_class_distribution(tmp_path):
    meta = _make_meta(tmp_path, n=2, label="bonafide") + _make_meta(tmp_path, n=3, label="spoof")
    ds = AudioDeepfakeDataset(meta, split="train", training=False)
    dist = get_class_distribution(ds)
    assert dist["bonafide"] == 2
    assert dist["spoof"] == 3
    assert dist["total"] == 5


# ── 16. Class weight calculation ─────────────────────────────
def test_class_weights(tmp_path):
    meta = _make_meta(tmp_path, n=2, label="bonafide") + _make_meta(tmp_path, n=8, label="spoof")
    ds = AudioDeepfakeDataset(meta, split="train", training=False)
    weights = get_class_weights(ds)
    assert weights.shape == torch.Size([2])
    # Bonafide is rarer → weight should be higher
    assert weights[0] > weights[1]
