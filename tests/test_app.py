import pytest
import torch
import numpy as np
import os
import shutil
import tempfile
import soundfile as sf
from src.predict import run_inference, load_frozen_model

@pytest.fixture(scope="module")
def device():
    return torch.device('cpu')

@pytest.fixture(scope="module")
def model(device):
    return load_frozen_model(device)

@pytest.fixture
def dummy_audio_path():
    path = "tests/dummy_audio.wav"
    sr = 16000
    # Generate 4 seconds of noise
    y = np.random.randn(sr * 4)
    sf.write(path, y, sr, format='WAV')
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_inference_valid_audio(dummy_audio_path, model, device):
    results = run_inference(dummy_audio_path, model, device, threshold=0.5)
    
    assert "spoof_probability" in results
    assert "bonafide_probability" in results
    assert "predicted_label" in results
    
    # Probability range check
    assert 0.0 <= results['spoof_probability'] <= 1.0
    assert 0.0 <= results['bonafide_probability'] <= 1.0
    assert np.isclose(results['spoof_probability'] + results['bonafide_probability'], 1.0)
    
    # Threshold behavior check
    if results['spoof_probability'] >= 0.5:
        assert results['predicted_label'] == 1
    else:
        assert results['predicted_label'] == 0

def test_inference_invalid_audio(model, device):
    with pytest.raises(ValueError):
        run_inference("non_existent_file.wav", model, device)

def test_gradcam_integration(dummy_audio_path, model, device):
    from src.explainability import generate_gradcam
    results = run_inference(dummy_audio_path, model, device)
    tensor_input = results['tensor_input']
    tensor_input.requires_grad = True
    
    # Target class 1
    cam = generate_gradcam(model, tensor_input, target_class=1)
    
    assert cam.shape == (128, 251)
    assert np.min(cam) >= 0.0
    assert np.max(cam) <= 1.0 or np.max(cam) == 0.0
