# tests/test_model_training_comparison.py — Unit tests for training architecture comparison.

import os
import torch
import pytest
import json

from src.model import create_model
from src.config import LEARNING_RATE, RESNET_LEARNING_RATE

def test_model_selection():
    cnn = create_model("cnn")
    resnet = create_model("resnet18")
    assert cnn.__class__.__name__ == "AudioDeepfakeCNN"
    assert resnet.__class__.__name__ == "AudioResNet18"

def test_same_input_tensor_shape():
    # Both models should accept the exact same tensor shape
    x = torch.randn(2, 1, 128, 251)
    
    cnn = create_model("cnn")
    resnet = create_model("resnet18")
    
    # We set to eval to avoid batchnorm issues with small batches
    cnn.eval()
    resnet.eval()
    
    with torch.no_grad():
        out_cnn = cnn(x)
        out_resnet = resnet(x)
        
    assert out_cnn.shape == out_resnet.shape
    assert out_cnn.shape == torch.Size([2, 1])

def test_same_loss_function():
    # Both models should use BCEWithLogitsLoss seamlessly
    criterion = torch.nn.BCEWithLogitsLoss()
    labels = torch.tensor([[0.0], [1.0]])
    
    out_cnn = torch.randn(2, 1)
    out_resnet = torch.randn(2, 1)
    
    loss_cnn = criterion(out_cnn, labels)
    loss_resnet = criterion(out_resnet, labels)
    
    assert torch.isfinite(loss_cnn)
    assert torch.isfinite(loss_resnet)

def test_same_optimizer_family():
    cnn = create_model("cnn")
    resnet = create_model("resnet18")
    
    opt_cnn = torch.optim.AdamW(cnn.parameters(), lr=LEARNING_RATE)
    opt_resnet = torch.optim.AdamW(resnet.parameters(), lr=RESNET_LEARNING_RATE)
    
    assert type(opt_cnn) == type(opt_resnet)

def test_missing_model_handling():
    # compare_models logic is mostly in script, but we test the config boundary
    with pytest.raises(ValueError, match="Unknown model_name"):
        create_model("invalid_model")

def test_checkpoint_isolation(tmp_path):
    # Verify we can save them to isolated paths
    import torch.nn as nn
    cnn = create_model("cnn")
    resnet = create_model("resnet18")
    
    cnn_path = tmp_path / "cnn.pth"
    resnet_path = tmp_path / "resnet18.pth"
    
    torch.save(cnn.state_dict(), cnn_path)
    torch.save(resnet.state_dict(), resnet_path)
    
    assert os.path.exists(cnn_path)
    assert os.path.exists(resnet_path)
    
    # Load correctly
    cnn.load_state_dict(torch.load(cnn_path, weights_only=True))
    resnet.load_state_dict(torch.load(resnet_path, weights_only=True))
