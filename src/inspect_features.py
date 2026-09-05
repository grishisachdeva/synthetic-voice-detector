# inspect_features.py — Generate representative Mel-spectrogram examples and print diagnostics.

import os
import csv
import numpy as np

from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram, extract_mfcc, plot_waveform, plot_mel_spectrogram
from src.config import SAMPLE_RATE


# Representative samples to inspect.
# Format: (file_path, label_hint, dataset_name)
SAMPLES = [
    # 2 BONAFIDE training
    ("data/raw/training/ASVspoof2019_LA_train/flac/LA_T_1000001.flac", "bonafide", "training"),
    ("data/raw/training/ASVspoof2019_LA_train/flac/LA_T_1000004.flac", "bonafide", "training"),
    # 2 SPOOF training
    ("data/raw/training/ASVspoof2019_LA_train/flac/LA_T_1000002.flac", "spoof", "training"),
    ("data/raw/training/ASVspoof2019_LA_train/flac/LA_T_1000003.flac", "spoof", "training"),
    # 1 BONAFIDE development
    ("data/raw/development/ASVspoof2019_LA_dev/flac/LA_D_2000001.flac", "bonafide", "development"),
    # 1 SPOOF development
    ("data/raw/development/ASVspoof2019_LA_dev/flac/LA_D_2000002.flac", "spoof", "development"),
    # 1 ASVspoof2021 DF evaluation
    ("data/raw/ASVspoof2021_DF/flac/DF_E_2000001.flac", "bonafide", "eval"),
]

PLOTS_DIR = "outputs/plots/features"


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("       MEL-SPECTROGRAM FEATURE INSPECTION")
    print("=" * 60)

    for path, label, dataset in SAMPLES:
        print(f"\n--- {path} ({label}, {dataset}) ---")

        if not os.path.exists(path):
            print(f"  SKIPPED: file not found.")
            continue

        # Preprocess
        result = preprocess_audio(path)
        waveform = result["waveform"]
        sr = result["sample_rate"]

        # Extract features
        mel_spec = extract_mel_spectrogram(waveform, sample_rate=sr)
        mfcc = extract_mfcc(waveform, sample_rate=sr)

        # Print diagnostics
        print(f"  Waveform shape   : {waveform.shape}")
        print(f"  Sample rate      : {sr}")
        print(f"  Duration         : {result['duration']} s")
        print(f"  Spectrogram shape: {mel_spec.shape}")
        print(f"  Spectrogram dtype: {mel_spec.dtype}")
        print(f"  Min value        : {mel_spec.min():.6f}")
        print(f"  Max value        : {mel_spec.max():.6f}")
        print(f"  Mean             : {mel_spec.mean():.6f}")
        print(f"  Std              : {mel_spec.std():.6f}")
        print(f"  Contains NaN     : {np.isnan(mel_spec).any()}")
        print(f"  Contains Inf     : {np.isinf(mel_spec).any()}")
        print(f"  MFCC shape       : {mfcc.shape}")

        # Save plots
        base = os.path.splitext(os.path.basename(path))[0]
        tag = f"{dataset}_{label}_{base}"

        plot_waveform(
            waveform, sr,
            os.path.join(PLOTS_DIR, f"{tag}_waveform.png"),
            title=f"Waveform — {base} ({label})"
        )
        plot_mel_spectrogram(
            mel_spec, sr,
            os.path.join(PLOTS_DIR, f"{tag}_mel_spectrogram.png"),
            title=f"Log-Mel Spectrogram — {base} ({label})"
        )

    print("\n" + "=" * 60)
    print("  Feature inspection complete. Plots saved to:", PLOTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
