import pytest
import os
import torch
import numpy as np
from src.model import AudioDeepfakeCNN
from src.explainability import generate_gradcam

@pytest.fixture
def dummy_model():
    model = AudioDeepfakeCNN()
    model.eval()
    return model

@pytest.fixture
def dummy_input():
    # Batch size 1, Channel 1, Height 128, Width 251
    tensor = torch.rand(1, 1, 128, 251)
    tensor.requires_grad = True
    return tensor

def test_gradcam_generation(dummy_model, dummy_input):
    cam = generate_gradcam(dummy_model, dummy_input, target_class=1)
    # Heatmap shape
    assert cam.shape == (128, 251)
    # Numerical range
    assert np.min(cam) >= 0.0
    assert np.max(cam) <= 1.0 or np.max(cam) == 0.0
    
def test_gradcam_negative_class(dummy_model, dummy_input):
    cam = generate_gradcam(dummy_model, dummy_input, target_class=0)
    assert cam.shape == (128, 251)
    
def test_gradcam_no_gradients():
    model = AudioDeepfakeCNN()
    # Missing layer mock
    model.block4 = None 
    tensor = torch.rand(1, 1, 128, 251)
    tensor.requires_grad = True
    cam = generate_gradcam(model, tensor, target_class=1)
    assert np.all(cam == 0.0)

def test_gradcam_reproducibility(dummy_model, dummy_input):
    # Fix seed for input if needed, but we already have a fixed dummy_input
    cam1 = generate_gradcam(dummy_model, dummy_input, target_class=1)
    
    # We must recreate the graph since we did a backward pass
    tensor2 = dummy_input.clone().detach()
    tensor2.requires_grad = True
    cam2 = generate_gradcam(dummy_model, tensor2, target_class=1)
    
    assert np.allclose(cam1, cam2, atol=1e-6)

def test_explainability_files_exist():
    assert os.path.exists("outputs/metrics/explainability_examples.csv")
    assert os.path.exists("outputs/metrics/explainability_summary.txt")
    assert os.path.exists("outputs/metrics/explainability_reproducibility.txt")
