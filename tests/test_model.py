# tests/test_model.py — Unit tests for the baseline CNN.

import torch
import pytest
import torch.nn as nn
import numpy as np

from src.model import AudioDeepfakeCNN, create_model, get_model_summary, validate_model_input, get_positive_class_weight
from src.config import INPUT_CHANNELS, NUM_CLASSES

def test_model_initialization():
    model = AudioDeepfakeCNN()
    assert isinstance(model, nn.Module)
    
def test_create_model():
    model = create_model()
    assert isinstance(model, nn.Module)
    # CPU should always be a valid fallback, or cuda if available
    device = next(model.parameters()).device
    assert device.type in ["cpu", "cuda"]

def test_parameter_count():
    model = AudioDeepfakeCNN()
    summary = get_model_summary(model)
    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] > 0
    assert summary["total_parameters"] == summary["trainable_parameters"] # All should be trainable initially

def test_forward_pass_and_shape():
    model = AudioDeepfakeCNN()
    model.eval() # Eval mode for shape check (no dropout)
    batch_size = 2
    # 4D tensor: [B, C, F, T]
    x = torch.randn(batch_size, 1, 128, 251)
    with torch.no_grad():
        logits = model(x)
        
    assert logits.shape == torch.Size([batch_size, NUM_CLASSES])
    
def test_output_dtype_and_validity():
    model = AudioDeepfakeCNN()
    model.eval()
    x = torch.randn(2, 1, 128, 251)
    with torch.no_grad():
        logits = model(x)
        
    assert logits.dtype == torch.float32
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_bce_logits_compatibility_and_backward():
    model = AudioDeepfakeCNN()
    model.train() # Train mode for backward pass
    x = torch.randn(2, 1, 128, 251)
    # Labels must match logits shape for BCEWithLogitsLoss
    labels = torch.tensor([[0.0], [1.0]])
    
    logits = model(x)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    
    assert torch.isfinite(loss)
    assert loss.dim() == 0 # scalar
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

def test_invalid_input_shape():
    # 3D instead of 4D
    x = torch.randn(1, 128, 251)
    with pytest.raises(ValueError, match="4-dimensional"):
        validate_model_input(x)
        
def test_invalid_channel_dim():
    # 3 channels instead of 1
    x = torch.randn(2, 3, 128, 251)
    with pytest.raises(ValueError, match="channel dimension must be 1"):
        validate_model_input(x)
        
def test_invalid_mel_dim():
    # 64 mel bands instead of 128
    x = torch.randn(2, 1, 64, 251)
    with pytest.raises(ValueError, match="Mel dimension must be 128"):
        validate_model_input(x)
        
def test_invalid_type():
    with pytest.raises(TypeError, match="PyTorch Tensor"):
        validate_model_input(np.random.rand(2, 1, 128, 251))
        
def test_get_positive_class_weight():
    counts = {"bonafide": 20, "spoof": 10}
    weight = get_positive_class_weight(counts)
    assert weight.item() == 2.0 # 20 / 10
    
    counts_zero = {"bonafide": 20, "spoof": 0}
    weight_zero = get_positive_class_weight(counts_zero)
    assert weight_zero.item() == 1.0 # fallback

def test_adaptive_pooling_handles_variable_time():
    """Verify that AdaptiveAvgPool2d allows different time dimensions."""
    model = AudioDeepfakeCNN()
    model.eval()
    
    # Standard length
    x1 = torch.randn(1, 1, 128, 251)
    # Shorter length (e.g. 2.0 seconds)
    x2 = torch.randn(1, 1, 128, 126)
    # Longer length (e.g. 6.0 seconds)
    x3 = torch.randn(1, 1, 128, 376)
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
        out3 = model(x3)
        
    assert out1.shape == torch.Size([1, 1])
    assert out2.shape == torch.Size([1, 1])
    assert out3.shape == torch.Size([1, 1])
