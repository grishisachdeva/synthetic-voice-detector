# tests/environment_check.py — Diagnostic script that prints the full
# development environment summary: versions, CUDA status, and device info.

import sys


def main():
    print("=" * 55)
    print("  Audio Deepfake Detector — Environment Diagnostic")
    print("=" * 55)
    print()

    # ── Python ──────────────────────────────────────────
    print(f"  Python version:       {sys.version.split()[0]}")

    # ── PyTorch ─────────────────────────────────────────
    try:
        import torch
        print(f"  PyTorch version:      {torch.__version__}")
    except ImportError:
        print("  PyTorch version:      NOT INSTALLED")

    try:
        import torchvision
        print(f"  Torchvision version:  {torchvision.__version__}")
    except ImportError:
        print("  Torchvision version:  NOT INSTALLED")

    try:
        import torchaudio
        print(f"  Torchaudio version:   {torchaudio.__version__}")
    except ImportError:
        print("  Torchaudio version:   NOT INSTALLED")

    # ── Audio / Data ────────────────────────────────────
    try:
        import librosa
        print(f"  Librosa version:      {librosa.__version__}")
    except ImportError:
        print("  Librosa version:      NOT INSTALLED")

    try:
        import numpy
        print(f"  NumPy version:        {numpy.__version__}")
    except ImportError:
        print("  NumPy version:        NOT INSTALLED")

    try:
        import pandas
        print(f"  Pandas version:       {pandas.__version__}")
    except ImportError:
        print("  Pandas version:       NOT INSTALLED")

    try:
        import sklearn
        print(f"  Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("  Scikit-learn version: NOT INSTALLED")

    # ── CUDA / GPU ──────────────────────────────────────
    print()
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available:       {cuda_available}")

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU name:             {gpu_name}")
            print(f"  CUDA version:         {torch.version.cuda}")
        else:
            print("  GPU name:             N/A (no CUDA GPU detected)")

        device = "cuda" if cuda_available else "cpu"
        print(f"  Device being used:    {device}")
    except Exception as e:
        print(f"  CUDA check failed:    {e}")

    print()
    print("=" * 55)
    print("  Diagnostic complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
