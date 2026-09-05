# features.py — Mel-spectrogram and MFCC feature extraction.

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from src.config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH,
    N_MELS, F_MIN, F_MAX, POWER, TOP_DB, N_MFCC
)
from src.preprocessing import preprocess_audio


def extract_mel_spectrogram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Convert a 1-D float32 waveform into a normalized log-Mel spectrogram.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 waveform (e.g. shape (64000,)).
    sample_rate : int
        Sampling rate of the waveform.

    Returns
    -------
    np.ndarray
        Normalized log-Mel spectrogram of shape (N_MELS, time_frames), dtype float32.
    """
    # --- Input validation ---
    if audio is None or len(audio) == 0:
        raise ValueError("Waveform is empty or None.")
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got {audio.ndim}-D array.")
    if np.isnan(audio).any():
        raise ValueError("Waveform contains NaN values.")
    if np.isinf(audio).any():
        raise ValueError("Waveform contains infinite values.")

    # --- Mel spectrogram (power) ---
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=POWER,
    )

    # --- Convert power to dB ---
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=TOP_DB)

    # --- Per-sample normalization: (X - mean) / (std + eps) ---
    eps = 1e-6
    mean = mel_spec_db.mean()
    std = mel_spec_db.std()
    mel_spec_norm = (mel_spec_db - mean) / (std + eps)

    # --- Final safety check ---
    mel_spec_norm = np.nan_to_num(mel_spec_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return mel_spec_norm.astype(np.float32)


def extract_mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract MFCC features from a 1-D float32 waveform.

    This is an OPTIONAL feature for future experimentation.
    The primary CNN input remains the log-Mel spectrogram.

    Returns
    -------
    np.ndarray
        MFCC matrix of shape (N_MFCC, time_frames), dtype float32.
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Waveform is empty or None.")
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got {audio.ndim}-D array.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX,
    )

    return mfcc.astype(np.float32)


def extract_features(audio_path: str) -> dict:
    """
    End-to-end feature extraction wrapper.

    Calls preprocess_audio() then extract_mel_spectrogram().

    Returns
    -------
    dict with keys: spectrogram, sample_rate, duration, original_path
    """
    result = preprocess_audio(audio_path)
    waveform = result["waveform"]
    sr = result["sample_rate"]

    spectrogram = extract_mel_spectrogram(waveform, sample_rate=sr)

    return {
        "spectrogram": spectrogram,
        "sample_rate": sr,
        "duration": result["duration"],
        "original_path": str(audio_path),
    }


# ============================================================
# Visualization helpers
# ============================================================

def plot_waveform(audio: np.ndarray, sample_rate: int, output_path: str, title: str = "Waveform") -> None:
    """Save a waveform plot to *output_path*."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration = len(audio) / sample_rate
    time_axis = np.linspace(0, duration, len(audio))

    plt.figure(figsize=(10, 3))
    plt.plot(time_axis, audio, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_mel_spectrogram(mel_spectrogram: np.ndarray, sample_rate: int, output_path: str,
                         title: str = "Log-Mel Spectrogram") -> None:
    """Save a Mel-spectrogram plot to *output_path*."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=F_MIN,
        fmax=F_MAX,
    )
    plt.colorbar(format="%+2.0f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
