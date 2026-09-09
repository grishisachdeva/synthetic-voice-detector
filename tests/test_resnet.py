# tests/test_resnet.py — Unit tests for ResNet18.

import torch
import pytest
import torch.nn as nn

from src.model import AudioResNet18, create_model, get_model_summary, validate_model_input
from src.config import INPUT_CHANNELS, NUM_CLASSES

def test_resnet_initialization():
    model = AudioResNet18(pretrained=False)
    assert isinstance(model, nn.Module)
    
def test_create_resnet_model():
    model = create_model(model_name="resnet18")
    assert isinstance(model, nn.Module)
    device = next(model.parameters()).device
    assert device.type in ["cpu", "cuda"]

def test_invalid_model_name():
    with pytest.raises(ValueError, match="Unknown model_name"):
        create_model(model_name="invalid")

def test_single_channel_input():
    model = AudioResNet18(pretrained=False)
    # The first conv should accept INPUT_CHANNELS
    assert model.resnet.conv1.in_channels == INPUT_CHANNELS
    
def test_parameter_count():
    model = AudioResNet18(pretrained=False)
    summary = get_model_summary(model)
    assert summary["total_parameters"] > 11000000 # Standard resnet18 is ~11M
    assert summary["pretrained_status"] is False

def test_forward_pass_and_shape():
    model = AudioResNet18(pretrained=False)
    model.eval()
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 251)
    with torch.no_grad():
        logits = model(x)
        
    assert logits.shape == torch.Size([batch_size, NUM_CLASSES])
    
def test_output_dtype_and_validity():
    model = AudioResNet18(pretrained=False)
    model.eval()
    x = torch.randn(2, 1, 128, 251)
    with torch.no_grad():
        logits = model(x)
        
    assert logits.dtype == torch.float32
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_bce_logits_compatibility_and_backward():
    model = AudioResNet18(pretrained=False)
    model.train()
    x = torch.randn(2, 1, 128, 251)
    labels = torch.tensor([[0.0], [1.0]])
    
    logits = model(x)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    
    assert torch.isfinite(loss)
    assert loss.dim() == 0
    
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

def test_input_validation():
    # Reuse validate_model_input
    x = torch.randn(2, 1, 128, 251)
    # Should pass without error
    validate_model_input(x)
    
    # 3 channels should fail
    x_bad = torch.randn(2, 3, 128, 251)
    with pytest.raises(ValueError, match="channel dimension must be 1"):
        validate_model_input(x_bad)
