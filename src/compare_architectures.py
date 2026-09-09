# compare_architectures.py — Model comparison utility.

import torch
from src.model import create_model, get_model_summary

def main():
    print("=" * 60)
    print("         ARCHITECTURE COMPARISON")
    print("=" * 60)

    # Dummy input
    x = torch.randn(4, 1, 128, 251)
    
    # CNN
    cnn = create_model(model_name="cnn")
    cnn_summary = get_model_summary(cnn)
    cnn_out = cnn(x.to(cnn_summary["device"]))
    
    print("\nMODEL: CNN")
    print(f"Input: {list(x.shape)}")
    print(f"Output: {list(cnn_out.shape)}")
    print(f"Parameters: {cnn_summary['total_parameters']}")
    
    # ResNet18
    resnet = create_model(model_name="resnet18")
    resnet_summary = get_model_summary(resnet)
    resnet_out = resnet(x.to(resnet_summary["device"]))
    
    print("\nMODEL: ResNet18")
    print(f"Input: {list(x.shape)}")
    print(f"Output: {list(resnet_out.shape)}")
    print(f"Parameters: {resnet_summary['total_parameters']}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
