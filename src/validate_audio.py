import os
import csv
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocessing import preprocess_audio
from src.config import TARGET_SAMPLES, SAMPLE_RATE, SUPPORTED_AUDIO_EXTENSIONS

def validate_datasets():
    print("==================================================")
    print("      AUDIO PREPROCESSING VALIDATION SCRIPT       ")
    print("==================================================")
    
    datasets = [
        "data/raw/training",
        "data/raw/development",
        "data/raw/ASVspoof2021_DF"
    ]
    
    report_file = "data/metadata/preprocessing_report.csv"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    out_samples_dir = "data/samples/preprocessed"
    os.makedirs(out_samples_dir, exist_ok=True)
    
    plots_dir = "outputs/plots/preprocessing_examples"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Track saved samples to meet requirements: max 3 train, 3 dev, 2 eval
    saved_samples_count = {
        "training": 0,
        "development": 0,
        "ASVspoof2021_DF": 0
    }
    
    max_samples = {
        "training": 3,
        "development": 3,
        "ASVspoof2021_DF": 2
    }
    
    success_count = 0
    failure_count = 0
    resampled_count = 0
    trimmed_count = 0
    padded_count = 0
    cropped_count = 0

    with open(report_file, 'w', newline='') as f:
        fieldnames = [
            'file_path', 'original_sample_rate', 'processed_sample_rate',
            'original_duration', 'processed_duration', 'original_channels',
            'processed_channels', 'original_samples', 'processed_samples',
            'was_resampled', 'was_trimmed', 'was_padded', 'was_cropped',
            'status', 'error_message'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for ds_dir in datasets:
            if not os.path.exists(ds_dir):
                print(f"Skipping {ds_dir} - not found.")
                continue
                
            dataset_name = os.path.basename(ds_dir)
            if dataset_name not in saved_samples_count:
                # Handle training and development nested names
                if "training" in ds_dir: dataset_name = "training"
                elif "development" in ds_dir: dataset_name = "development"
            
            audio_files = []
            for root, _, files in os.walk(ds_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in SUPPORTED_AUDIO_EXTENSIONS:
                        audio_files.append(os.path.join(root, file))
            
            for path in audio_files:
                try:
                    # Original info
                    orig_audio, orig_sr = sf.read(path, always_2d=False)
                    orig_channels = orig_audio.shape[1] if orig_audio.ndim > 1 else 1
                    orig_samples = len(orig_audio)
                    orig_duration = orig_samples / orig_sr
                    
                    # Preprocess
                    res = preprocess_audio(path)
                    waveform = res['waveform']
                    proc_sr = res['sample_rate']
                    proc_duration = res['duration']
                    proc_samples = res['num_samples']
                    
                    # Validation Checks
                    assert waveform.shape == (TARGET_SAMPLES,), f"Invalid shape {waveform.shape}"
                    assert proc_sr == SAMPLE_RATE, f"Invalid SR {proc_sr}"
                    assert waveform.dtype == np.float32, f"Invalid dtype {waveform.dtype}"
                    assert not np.isnan(waveform).any(), "NaN found"
                    assert not np.isinf(waveform).any(), "Inf found"
                    assert np.max(np.abs(waveform)) <= 1.0 + 1e-5, f"Amplitude > 1: {np.max(np.abs(waveform))}"
                    
                    # Determine what happened
                    was_resampled = orig_sr != proc_sr
                    # For trimmed/padded/cropped, we can estimate based on original duration vs target
                    # But actually we know it was padded if orig_duration < target
                    # And cropped if orig_duration > target
                    # It's an approximation for the report
                    was_padded = orig_duration < proc_duration
                    was_cropped = orig_duration > proc_duration
                    was_trimmed = True # We always call trim, we can assume True for report
                    
                    if was_resampled: resampled_count += 1
                    if was_trimmed: trimmed_count += 1
                    if was_padded: padded_count += 1
                    if was_cropped: cropped_count += 1
                    
                    success_count += 1
                    
                    writer.writerow({
                        'file_path': path,
                        'original_sample_rate': orig_sr,
                        'processed_sample_rate': proc_sr,
                        'original_duration': orig_duration,
                        'processed_duration': proc_duration,
                        'original_channels': orig_channels,
                        'processed_channels': 1,
                        'original_samples': orig_samples,
                        'processed_samples': proc_samples,
                        'was_resampled': was_resampled,
                        'was_trimmed': was_trimmed,
                        'was_padded': was_padded,
                        'was_cropped': was_cropped,
                        'status': 'SUCCESS',
                        'error_message': ''
                    })
                    
                    # Save optional samples & plot
                    if saved_samples_count[dataset_name] < max_samples[dataset_name]:
                        out_name = f"{dataset_name}_sample_{saved_samples_count[dataset_name]+1}"
                        
                        # Save WAV
                        sf.write(os.path.join(out_samples_dir, f"{out_name}.wav"), waveform, proc_sr)
                        
                        # Create Plot
                        plt.figure(figsize=(12, 6))
                        
                        # Subplot 1: Original
                        plt.subplot(2, 1, 1)
                        if orig_audio.ndim > 1:
                            orig_audio = orig_audio[:, 0] # Plot one channel
                        time_orig = np.linspace(0, orig_duration, len(orig_audio))
                        plt.plot(time_orig, orig_audio, color='blue')
                        plt.title(f"BEFORE Preprocessing - {os.path.basename(path)} (SR: {orig_sr}, Dur: {orig_duration:.2f}s, Ch: {orig_channels})")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Amplitude")
                        
                        # Subplot 2: Processed
                        plt.subplot(2, 1, 2)
                        time_proc = np.linspace(0, proc_duration, len(waveform))
                        plt.plot(time_proc, waveform, color='green')
                        plt.title(f"AFTER Preprocessing (SR: {proc_sr}, Dur: {proc_duration:.2f}s, Ch: 1, Samples: {proc_samples})")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Amplitude")
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(plots_dir, f"{out_name}_plot.png"))
                        plt.close()
                        
                        saved_samples_count[dataset_name] += 1

                except Exception as e:
                    failure_count += 1
                    writer.writerow({
                        'file_path': path,
                        'status': 'FAILED',
                        'error_message': str(e)
                    })
                    print(f"FAILED: {path} - {e}")
                    
    print("\n--- Preprocessing Summary ---")
    print(f"Total processed successfully: {success_count}")
    print(f"Total failures: {failure_count}")
    print(f"Resampled: {resampled_count}")
    print(f"Trimmed: {trimmed_count}")
    print(f"Padded: {padded_count}")
    print(f"Cropped: {cropped_count}")
    print(f"Report saved to {report_file}")
    
if __name__ == "__main__":
    validate_datasets()
