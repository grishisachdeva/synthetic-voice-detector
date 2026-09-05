# tests/test_features.py — Unit tests for Mel-spectrogram feature extraction.

import numpy as np
import pytest
import soundfile as sf

from src.config import SAMPLE_RATE, TARGET_SAMPLES, N_MELS, N_MFCC, HOP_LENGTH
from src.features import extract_mel_spectrogram, extract_mfcc, extract_features


def _sine_wave(duration: float = 4.0, sr: int = SAMPLE_RATE, freq: float = 440.0) -> np.ndarray:
    """Generate a float32 sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# Expected time frames for 64000 samples with hop_length=256
EXPECTED_TIME_FRAMES = 1 + (TARGET_SAMPLES // HOP_LENGTH)  # librosa convention


# ── 1. Mel spectrogram shape ──────────────────────────────────
def test_mel_spectrogram_shape():
    audio = _sine_wave()
    spec = extract_mel_spectrogram(audio)
    assert spec.shape[0] == N_MELS
    assert spec.shape[1] == EXPECTED_TIME_FRAMES
    print(f"Spectrogram shape: {spec.shape}")


# ── 2. Frequency dimension equals 128 ────────────────────────
def test_frequency_dimension():
    spec = extract_mel_spectrogram(_sine_wave())
    assert spec.shape[0] == 128


# ── 3. Correct dtype ─────────────────────────────────────────
def test_dtype():
    spec = extract_mel_spectrogram(_sine_wave())
    assert spec.dtype == np.float32


# ── 4. No NaN ────────────────────────────────────────────────
def test_no_nan():
    spec = extract_mel_spectrogram(_sine_wave())
    assert not np.isnan(spec).any()


# ── 5. No Inf ────────────────────────────────────────────────
def test_no_inf():
    spec = extract_mel_spectrogram(_sine_wave())
    assert not np.isinf(spec).any()


# ── 6. Deterministic output ──────────────────────────────────
def test_deterministic():
    audio = _sine_wave()
    s1 = extract_mel_spectrogram(audio.copy())
    s2 = extract_mel_spectrogram(audio.copy())
    np.testing.assert_array_equal(s1, s2)


# ── 7. Different signals produce valid outputs ───────────────
def test_different_signals():
    s1 = extract_mel_spectrogram(_sine_wave(freq=200.0))
    s2 = extract_mel_spectrogram(_sine_wave(freq=2000.0))
    assert s1.shape == s2.shape
    assert not np.array_equal(s1, s2)


# ── 8. Short/edge-case input ─────────────────────────────────
def test_short_input():
    short = np.ones(256, dtype=np.float32) * 0.1
    spec = extract_mel_spectrogram(short)
    assert spec.shape[0] == N_MELS
    assert spec.dtype == np.float32
    assert not np.isnan(spec).any()


# ── 9. Silent audio ──────────────────────────────────────────
def test_silent_audio():
    silent = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    spec = extract_mel_spectrogram(silent)
    assert spec.shape[0] == N_MELS
    assert not np.isnan(spec).any()
    assert not np.isinf(spec).any()


# ── 10. MFCC extraction shape ────────────────────────────────
def test_mfcc_shape():
    audio = _sine_wave()
    mfcc = extract_mfcc(audio)
    assert mfcc.shape[0] == N_MFCC
    assert mfcc.dtype == np.float32


# ── 11. extract_features uses preprocessing pipeline ─────────
def test_extract_features_pipeline(tmp_path):
    path = tmp_path / "test.wav"
    audio = _sine_wave(duration=5.0, sr=8000)
    sf.write(str(path), audio, 8000)

    result = extract_features(str(path))

    assert "spectrogram" in result
    assert "sample_rate" in result
    assert "duration" in result
    assert "original_path" in result

    spec = result["spectrogram"]
    assert spec.shape[0] == N_MELS
    assert spec.shape[1] == EXPECTED_TIME_FRAMES
    assert spec.dtype == np.float32
    assert result["sample_rate"] == SAMPLE_RATE


# ── Input validation ─────────────────────────────────────────
def test_empty_waveform_raises():
    with pytest.raises(ValueError, match="empty"):
        extract_mel_spectrogram(np.array([], dtype=np.float32))


def test_nan_waveform_raises():
    audio = np.array([0.0, np.nan, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        extract_mel_spectrogram(audio)


def test_inf_waveform_raises():
    audio = np.array([0.0, np.inf, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="infinite"):
        extract_mel_spectrogram(audio)


def test_2d_waveform_raises():
    audio = np.ones((2, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        extract_mel_spectrogram(audio)
