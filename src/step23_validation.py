import os
import json
import hashlib
import numpy as np
import soundfile as sf
import tempfile
from src.predict import run_inference, load_frozen_model
import torch

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_integrity():
    files = [
        "models/cnn_real/best_cnn_real.pth",
        "outputs/metrics/final_results.json",
        "outputs/metrics/final_frozen_configuration.json",
        "outputs/predictions/final_evaluation/final_df_predictions.csv"
    ]
    hashes = {}
    for f in files:
        if os.path.exists(f):
            hashes[f] = compute_md5(f)
        else:
            hashes[f] = "MISSING"
    return hashes

def test_edge_cases():
    print("Running Edge Case Validation...")
    device = torch.device('cpu')
    model = load_frozen_model(device)
    
    # 1. Unsupported file
    with open("dummy.txt", "w") as f:
        f.write("not an audio file")
    try:
        run_inference("dummy.txt", model, device)
        print("FAIL: Unsupported file did not raise exception.")
    except Exception as e:
        print("PASS: Unsupported file raised exception properly.")
    finally:
        os.remove("dummy.txt")
        
    # 2. Silent audio
    sf.write("silent.wav", np.zeros(16000 * 4), 16000)
    try:
        run_inference("silent.wav", model, device)
        print("PASS: Silent audio processed gracefully.")
    except Exception as e:
        print(f"FAIL: Silent audio crashed: {e}")
    finally:
        os.remove("silent.wav")
        
    # 3. Very short audio (0.1s)
    sf.write("short.wav", np.random.randn(1600), 16000)
    try:
        run_inference("short.wav", model, device)
        print("PASS: Short audio processed gracefully (padded).")
    except Exception as e:
        print(f"FAIL: Short audio crashed: {e}")
    finally:
        os.remove("short.wav")
        
    # 4. Long audio (10s)
    sf.write("long.wav", np.random.randn(160000), 16000)
    try:
        run_inference("long.wav", model, device)
        print("PASS: Long audio processed gracefully (truncated).")
    except Exception as e:
        print(f"FAIL: Long audio crashed: {e}")
    finally:
        os.remove("long.wav")

if __name__ == "__main__":
    h1 = check_integrity()
    print("Initial Hashes:")
    for k, v in h1.items():
        print(f"{k}: {v}")
    
    test_edge_cases()
    
    h2 = check_integrity()
    if h1 == h2:
        print("\nIntegrity check PASSED. No files were modified.")
    else:
        print("\nIntegrity check FAILED. Files were modified.")
