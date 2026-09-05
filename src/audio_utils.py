import os
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
from src.config import SUPPORTED_AUDIO_EXTENSIONS

def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio to mono by averaging channels.
    Assumes audio shape is (channels, samples) or (samples,).
    """
    if audio.ndim == 1:
        return audio.astype(np.float32)
    elif audio.ndim == 2:
        # If shape is (samples, channels), transpose it
        # librosa and soundfile conventions vary, but soundfile returns (frames, channels)
        # Let's handle both. We expect (samples, channels) from soundfile.
        # Actually librosa expects (channels, samples) or (samples,).
        # We will average across the smaller dimension assuming it's channels.
        if audio.shape[0] < audio.shape[1]:
            # shape is (channels, samples)
            mono_audio = np.mean(audio, axis=0)
        else:
            # shape is (samples, channels)
            mono_audio = np.mean(audio, axis=1)
        return mono_audio.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio array dimension: {audio.ndim}")

def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample audio to target_sr if it differs from original_sr.
    """
    if original_sr == target_sr:
        return audio.astype(np.float32)
    
    # Use librosa for high-quality resampling
    # librosa.resample expects shape (..., samples)
    resampled = librosa.resample(y=audio, orig_sr=original_sr, target_sr=target_sr)
    return resampled.astype(np.float32)

def load_audio(audio_path, target_sr=16000, mono=True):
    """
    Load audio, optionally convert to mono, and resample.
    Returns (waveform, sample_rate)
    """
    path = Path(audio_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
        
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio extension: {path.suffix}. Supported: {SUPPORTED_AUDIO_EXTENSIONS}")
        
    try:
        # Load audio using soundfile. 
        # soundfile returns audio as (frames, channels) or (frames,)
        audio, sr = sf.read(str(path), dtype='float32')
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {path}. It may be corrupted. Error: {e}")
        
    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path}")

    # Convert to mono
    if mono:
        audio = convert_to_mono(audio)
        
    # Resample
    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
        
    return audio, sr
