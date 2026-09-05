import numpy as np
import librosa
from src.config import (
    SAMPLE_RATE, TARGET_DURATION, TARGET_SAMPLES, 
    SILENCE_TOP_DB, NORMALIZE_AUDIO, CROP_MODE, PAD_MODE
)
from src.audio_utils import load_audio

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Numerically stable amplitude normalization.
    """
    if np.isnan(audio).any() or np.isinf(audio).any():
        raise ValueError("Audio contains NaN or infinite values before normalization.")
        
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-8:
        audio = audio / max_amp
        
    if np.isnan(audio).any() or np.isinf(audio).any():
        raise ValueError("Audio contains NaN or infinite values after normalization.")
        
    return audio.astype(np.float32)

def trim_silence(audio: np.ndarray, top_db: float = 30) -> np.ndarray:
    """
    Remove leading and trailing silence using librosa.
    Returns the trimmed audio. If completely silent, returns a small empty array.
    """
    if len(audio) == 0:
        return audio
        
    if np.max(np.abs(audio)) < 1e-8:
        return np.zeros(1, dtype=np.float32)
        
    # Trim silence
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    
    if len(trimmed_audio) == 0:
        # Fallback for completely silent audio - return 1 sample of silence to avoid crashes
        return np.zeros(1, dtype=np.float32)
        
    return trimmed_audio.astype(np.float32)

def pad_or_crop(audio: np.ndarray, target_samples: int, mode: str = "center", pad_mode: str = "constant") -> np.ndarray:
    """
    Pad or crop audio to exactly target_samples length.
    """
    length = len(audio)
    
    if length == target_samples:
        return audio
        
    if length < target_samples:
        # Pad
        pad_length = target_samples - length
        if mode == "center":
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            padded = np.pad(audio, (pad_left, pad_right), mode=pad_mode)
        else: # e.g., 'right'
            padded = np.pad(audio, (0, pad_length), mode=pad_mode)
        return padded.astype(np.float32)
        
    else:
        # Crop
        crop_length = length - target_samples
        if mode == "center":
            start = crop_length // 2
            cropped = audio[start : start + target_samples]
        else: # e.g., 'right'
            cropped = audio[:target_samples]
        return cropped.astype(np.float32)

def preprocess_audio(audio_path: str) -> dict:
    """
    Complete preprocessing pipeline.
    """
    # 1. Load, convert to mono, resample
    audio, sr = load_audio(audio_path, target_sr=SAMPLE_RATE, mono=True)
    
    # 2. Normalize
    if NORMALIZE_AUDIO:
        audio = normalize_audio(audio)
        
    # 3. Trim Silence
    audio = trim_silence(audio, top_db=SILENCE_TOP_DB)
    
    # 4. Pad or Crop
    audio = pad_or_crop(audio, TARGET_SAMPLES, mode=CROP_MODE, pad_mode=PAD_MODE)
    
    # 5. Final validation
    if np.isnan(audio).any() or np.isinf(audio).any():
        raise ValueError("NaN or inf encountered in final preprocessed waveform.")
        
    return {
        "waveform": audio,
        "sample_rate": SAMPLE_RATE,
        "duration": TARGET_DURATION,
        "num_samples": TARGET_SAMPLES
    }
