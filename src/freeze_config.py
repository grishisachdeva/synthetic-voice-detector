import json
import os
from src.config import (
    SAMPLE_RATE, TARGET_DURATION, N_FFT, HOP_LENGTH,
    WIN_LENGTH, N_MELS, F_MIN, F_MAX, NORMALIZE_AUDIO
)

def freeze_configuration():
    # Read selection report
    with open('outputs/metrics/final_model_selection.json', 'r') as f:
        selection = json.load(f)
        
    frozen_config = {
        "model_name": selection["selected_model"],
        "checkpoint": selection["selected_checkpoint"],
        "sample_rate": SAMPLE_RATE,
        "target_duration": TARGET_DURATION,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "n_mels": N_MELS,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "normalization_method": "max_amp_stable" if NORMALIZE_AUDIO else "none",
        "classification_threshold": selection["selection_metrics"]["threshold"],
        "dataset_training": "ASVspoof 2019 LA Training (25,380 files)",
        "dataset_development": "ASVspoof 2019 LA Development (24,844 files)",
        "dataset_evaluation": "ASVspoof 2021 DF Evaluation (Subset)"
    }
    
    with open('outputs/metrics/final_frozen_configuration.json', 'w') as f:
        json.dump(frozen_config, f, indent=4)
        
    print("Configuration frozen.")

if __name__ == "__main__":
    freeze_configuration()
