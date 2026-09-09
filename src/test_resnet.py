# test_resnet.py — Forward pass and loss smoke test for ResNet18.

import os
import torch
import torch.nn as nn
from src.model import create_model, get_model_summary, validate_model_input

def main():
    print("=" * 60)
    print("         RESNET18 SMOKE TEST")
    print("=" * 60)

    # 1. Create model
    model = create_model(model_name="resnet18")
    
    # Generate model summary report
    summary = get_model_summary(model)
    print("\n--- Model Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
        
    # Save model summary to report
    os.makedirs("outputs/metrics", exist_ok=True)
    report_path = "outputs/metrics/resnet18_model_summary.txt"
    with open(report_path, "w") as f:
        f.write("ResNet18 Architecture Report\n")
        f.write("============================\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nLayer-by-layer Architecture:\n")
        f.write(str(model))
    print(f"\nModel architecture report saved to: {report_path}")

    # 2. Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 251, device=summary["device"])
    print(f"\n--- Forward Pass Test ---")
    print(f"  Input shape : {x.shape}")
    print(f"  Input dtype : {x.dtype}")
    
    # Validate input using our safe check
    try:
        validate_model_input(x)
        print("  Input validation: PASS")
    except Exception as e:
        print(f"  Input validation: FAIL -> {e}")
        return

    # 3. Run forward pass
    model.train() 
    logits = model(x)
    print(f"  Output shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")
    
    # Verify output constraints
    assert logits.shape == torch.Size([batch_size, 1]), "Output shape mismatch!"
    assert logits.dtype == torch.float32, "Output dtype mismatch!"
    assert not torch.isnan(logits).any(), "Output contains NaN!"
    assert not torch.isinf(logits).any(), "Output contains Inf!"
    print("  Output validation: PASS (No NaN/Inf, correct shape/dtype)")

    # Loss Test
    print(f"\n--- Loss & Backward Pass Test ---")
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], device=summary["device"]).unsqueeze(1)
    
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    print(f"  Loss value  : {loss.item():.6f}")
    
    assert torch.isfinite(loss), "Loss is not finite!"
    
    # Backward pass
    loss.backward()
    print("  Backward pass: PASS")
    
    # Verify gradients
    has_grad = False
    valid_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"    ERROR: Invalid gradient in layer: {name}")
                    valid_grad = False
            else:
                print(f"    WARNING: No gradient for layer: {name}")
                
    if has_grad and valid_grad:
        print("  Gradient check: PASS (Gradients are present and valid)")
    else:
        print("  Gradient check: FAIL")

    print("\n" + "=" * 60)
    print("  Smoke test complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
