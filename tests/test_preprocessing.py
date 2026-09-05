import numpy as np
import pytest
import soundfile as sf
import os
from pathlib import Path
from src.config import TARGET_SAMPLES, SAMPLE_RATE
from src.audio_utils import convert_to_mono, resample_audio, load_audio
from src.preprocessing import normalize_audio, trim_silence, pad_or_crop, preprocess_audio

@pytest.fixture
def temp_audio_dir(tmp_path):
    d = tmp_path / "audio"
    d.mkdir()
    return d

def generate_sine_wave(duration, sr, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_convert_to_mono():
    # 1. mono audio
    mono = np.ones(100, dtype=np.float32)
    res = convert_to_mono(mono)
    assert res.shape == (100,)
    
    # 2. stereo -> mono
    stereo = np.ones((100, 2), dtype=np.float32)
    stereo[:, 1] = 0.0 # avg should be 0.5
    res = convert_to_mono(stereo)
    assert res.shape == (100,)
    assert np.allclose(res, 0.5)

def test_resample_audio():
    # 3. resampling
    audio = generate_sine_wave(1.0, 8000)
    resampled = resample_audio(audio, 8000, 16000)
    assert len(resampled) == 16000
    
    # original_sr == target_sr
    same = resample_audio(audio, 8000, 8000)
    assert len(same) == 8000
    assert np.allclose(audio, same)

def test_normalize_audio():
    # 4. normalization
    audio = np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32)
    norm = normalize_audio(audio)
    assert np.max(np.abs(norm)) == 1.0
    assert np.allclose(norm, np.array([0.0, 0.25, -0.5, 1.0]))

def test_trim_silence():
    # 5. silence trimming
    audio = np.concatenate([np.zeros(16000), np.ones(16000), np.zeros(16000)]).astype(np.float32)
    trimmed = trim_silence(audio)
    # the exact length depends on librosa's frame length, but it should be shorter
    assert len(trimmed) < len(audio)
    assert np.max(trimmed) > 0 # speech is preserved
    
    # 9. silent audio
    silent = np.zeros(16000, dtype=np.float32)
    trimmed_silent = trim_silence(silent)
    assert len(trimmed_silent) == 1
    assert trimmed_silent[0] == 0.0

def test_pad_or_crop():
    # 6. short audio padding
    short = np.ones(100, dtype=np.float32)
    padded = pad_or_crop(short, 200, mode="center")
    assert len(padded) == 200
    assert padded[50] == 1.0
    assert padded[0] == 0.0 # constant pad default is 0
    
    # 7. long audio cropping
    long = np.ones(300, dtype=np.float32)
    cropped = pad_or_crop(long, 200, mode="center")
    assert len(cropped) == 200
    
    # 8. exact-length audio
    exact = np.ones(200, dtype=np.float32)
    same = pad_or_crop(exact, 200)
    assert len(same) == 200

def test_nan_and_inf_protection():
    # 10. NaN protection
    audio_nan = np.array([0.0, np.nan, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        normalize_audio(audio_nan)
        
    # 11. infinity protection
    audio_inf = np.array([0.0, np.inf, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="infinite"):
        normalize_audio(audio_inf)

def test_file_loading_exceptions(temp_audio_dir):
    # 12. corrupt-file handling
    corrupt_path = temp_audio_dir / "corrupt.wav"
    corrupt_path.write_bytes(b"not a valid wav file")
    with pytest.raises(RuntimeError, match="corrupted"):
        load_audio(str(corrupt_path))
        
    # 13. unsupported-extension handling
    unsupported_path = temp_audio_dir / "test.txt"
    unsupported_path.write_bytes(b"dummy text")
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        load_audio(str(unsupported_path))

def test_full_preprocessing_pipeline(temp_audio_dir):
    # Setup a valid file
    valid_path = temp_audio_dir / "valid.wav"
    audio_data = generate_sine_wave(5.0, 8000) # 5 seconds, 8kHz
    sf.write(str(valid_path), audio_data, 8000)
    
    res = preprocess_audio(str(valid_path))
    
    # 14. final waveform shape
    assert res['waveform'].shape == (TARGET_SAMPLES,)
    
    # 15. final sample rate
    assert res['sample_rate'] == SAMPLE_RATE
    
    assert res['duration'] == 4.0
    assert res['num_samples'] == TARGET_SAMPLES
    assert res['waveform'].dtype == np.float32
