# model.py — Baseline Convolutional Neural Network architecture and utilities.

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import (
    INPUT_CHANNELS, NUM_CLASSES, DROPOUT, N_MELS,
    RESNET_DROPOUT, RESNET_PRETRAINED
)

class AudioDeepfakeCNN(nn.Module):
    """
    Baseline CNN for Audio Deepfake Detection from Log-Mel Spectrograms.
    
    Expected Input:
        x: Tensor of shape (batch_size, 1, 128, time_frames)
        where 1 is the channel dimension, 128 is the number of Mel bands.
        
    Expected Output:
        logits: Tensor of shape (batch_size, 1)
        Raw unnormalized scores (for BCEWithLogitsLoss).
    """
    def __init__(self, in_channels: int = INPUT_CHANNELS, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super(AudioDeepfakeCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            # No MaxPool here, handled by AdaptiveAvgPool2d next
        )
        
        # Global Pooling
        # Reduces any spatial dimensions (H, W) down to 1x1 per channel.
        # This makes the model robust to varying time lengths and fixed frequency dimension.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=num_classes)
            # NO Sigmoid here! We output raw logits for BCEWithLogitsLoss.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


class AudioResNet18(nn.Module):
    """
    ResNet18 adapted for 1-channel Log-Mel Spectrogram inputs.
    """
    def __init__(self, in_channels: int = INPUT_CHANNELS, num_classes: int = NUM_CLASSES, 
                 pretrained: bool = RESNET_PRETRAINED, dropout: float = RESNET_DROPOUT):
        super(AudioResNet18, self).__init__()
        
        self.pretrained_status = pretrained
        
        if pretrained:
            try:
                # Use the modern weights API if available
                weights = models.ResNet18_Weights.DEFAULT
                self.resnet = models.resnet18(weights=weights)
            except Exception as e:
                raise RuntimeError(f"Failed to load pretrained weights for ResNet18: {e}. "
                                   "Network restrictions might prevent downloading weights.")
        else:
            self.resnet = models.resnet18(weights=None)
            
        # 1. Modify the first convolutional layer to accept 1 channel instead of 3
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # 2. Modify the fully connected layer for binary classification
        # Original: Linear(in_features=512, out_features=1000, bias=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
            # NO Sigmoid here! We output raw logits for BCEWithLogitsLoss.
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.resnet(x)
        return logits


def create_model(model_name: str = "cnn", config=None) -> nn.Module:
    """
    Factory function to instantiate a model ("cnn" or "resnet18") and move it to the optimal device.
    
    Returns
    -------
    nn.Module
        The initialized model on CUDA (if available) or CPU.
    """
    if model_name.lower() == "cnn":
        model = AudioDeepfakeCNN()
    elif model_name.lower() == "resnet18":
        model = AudioResNet18()
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'cnn' or 'resnet18'.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def get_model_summary(model: nn.Module) -> dict:
    """
    Returns a simple dictionary containing model summary statistics.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Check device of the first parameter to infer model device
    device = next(model.parameters()).device if list(model.parameters()) else "unknown"
    
    summary = {
        "model_name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_shape": "(batch_size, 1, 128, time_frames)",
        "output_shape": "(batch_size, 1)",
        "device": str(device)
    }
    
    if hasattr(model, 'pretrained_status'):
        summary["pretrained_status"] = model.pretrained_status
        
    return summary


def get_positive_class_weight(class_counts: dict) -> torch.Tensor:
    """
    Calculates the positive class weight for binary classification using BCEWithLogitsLoss.
    
    Formula: pos_weight = negative_samples / positive_samples
    
    In our dataset schema:
    - BONAFIDE (0) is the negative class.
    - SPOOF (1) is the positive class.
    
    Parameters
    ----------
    class_counts : dict
        A dictionary with keys 'bonafide' and 'spoof' containing their respective counts.
        
    Returns
    -------
    torch.Tensor
        A 1D tensor containing the scalar pos_weight.
    """
    num_bonafide = class_counts.get("bonafide", 0)
    num_spoof = class_counts.get("spoof", 0)
    
    if num_spoof == 0:
        # Avoid division by zero if there are no spoof samples. 
        # Fallback to weight 1.0 (no weighting).
        weight = 1.0
    else:
        weight = float(num_bonafide) / float(num_spoof)
        
    return torch.tensor([weight], dtype=torch.float32)


def validate_model_input(x: torch.Tensor) -> None:
    """
    Validates that a tensor matches the expected model input requirements.
    
    Expected shape: [batch_size, 1, 128, 251]
    (where 251 is the target time frames from 4.0s at 16kHz with 256 hop, 
    but we mainly care about 4D, 1 channel, 128 Mel bands).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch Tensor, got {type(x)}.")
        
    if x.ndim != 4:
        raise ValueError(f"Input must be 4-dimensional (B, C, F, T). Got {x.ndim} dimensions.")
        
    batch, channels, mels, time_frames = x.shape
    
    if channels != INPUT_CHANNELS:
        raise ValueError(f"Input channel dimension must be {INPUT_CHANNELS}. Got {channels}.")
        
    if mels != N_MELS:
        raise ValueError(f"Input Mel dimension must be {N_MELS}. Got {mels}.")
        
    # Technically time_frames depends on audio duration and hop length.
    # Our preprocessing forces 64000 samples -> 251 frames.
    if time_frames != 251:
        raise ValueError(f"Input time dimension must be 251 for 4.0s audio. Got {time_frames}.")
        
    if not x.is_floating_point():
        raise TypeError(f"Input tensor must have a floating point dtype. Got {x.dtype}.")
