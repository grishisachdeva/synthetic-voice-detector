import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram

CACHE_ROOT = Path("data/processed/mel_cache")
METADATA_DIR = CACHE_ROOT / "metadata"

def validate_feature(feature_array):
    if not isinstance(feature_array, np.ndarray):
        return False, "Not a numpy array"
    if feature_array.shape != (128, 251):
        return False, f"Invalid shape: {feature_array.shape}"
    if feature_array.dtype != np.float32:
        return False, f"Invalid dtype: {feature_array.dtype}"
    if np.isnan(feature_array).any():
        return False, "Contains NaN"
    if np.isinf(feature_array).any():
        return False, "Contains Inf"
    return True, "OK"

def main():
    parser = argparse.ArgumentParser(description="Build Mel-spectrogram numerical cache.")
    parser.add_argument("--partition", type=str, required=True, choices=["training", "development"], help="Partition to cache")
    parser.add_argument("--resume", action="store_true", help="Skip valid existing cache files")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files to process (for validation/testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print estimates and exit without writing files")
    args = parser.parse_args()

    partition = args.partition
    metadata_path = Path(f"data/metadata/{partition}_metadata.csv")
    
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        return

    df = pd.read_csv(metadata_path)
    if args.max_files is not None:
        df = df.head(args.max_files)
        
    total_files = len(df)
    
    # Dry Run
    if args.dry_run:
        bytes_per_feature = 128512 # 128 * 251 * 4 bytes + ~128 bytes header
        total_bytes = total_files * bytes_per_feature
        print(f"--- DRY RUN: {partition.upper()} ---")
        print(f"Files to process: {total_files}")
        print(f"Estimated output count: {total_files}")
        print(f"Bytes per feature: {bytes_per_feature}")
        print(f"Estimated disk usage: {total_bytes / (1024**3):.2f} GB")
        print("First few source paths:")
        for idx in range(min(3, total_files)):
            print(f"  {df.iloc[idx]['file_path']}")
        print("First few output paths:")
        for idx in range(min(3, total_files)):
            out_path = CACHE_ROOT / partition / f"{df.iloc[idx]['file_id']}.npy"
            print(f"  {out_path}")
        return

    # Setup tracking
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.time()
    
    cache_entries = []
    failures = []
    
    out_dir = CACHE_ROOT / partition
    out_dir.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_csv_path = METADATA_DIR / f"{partition}_cache.csv"
    failures_csv_path = METADATA_DIR / "cache_failures.csv"

    # Pre-load existing failures if we are resuming to append (or just overwrite)
    # We will write/overwrite for the current partition. 
    # To keep it simple, we collect failures and append later.

    print(f"Starting {partition} cache generation for {total_files} files...")
    
    for idx, row in df.iterrows():
        file_id = row['file_id']
        source_path = row['file_path']
        cache_path = out_dir / f"{file_id}.npy"
        
        # Resume Check
        needs_compute = True
        if args.resume and cache_path.exists():
            try:
                existing_arr = np.load(cache_path)
                valid, msg = validate_feature(existing_arr)
                if valid:
                    needs_compute = False
                    skipped_count += 1
                    status = "skipped_valid"
                else:
                    # Invalid cache file, needs recompute
                    needs_compute = True
            except Exception:
                # Corrupt file, needs recompute
                needs_compute = True
                
        if needs_compute:
            try:
                pre_dict = preprocess_audio(source_path)
                waveform = pre_dict['waveform']
                
                # Extract natively as 1-D (it expects a numpy array)
                mel_array = extract_mel_spectrogram(waveform)
                
                valid, msg = validate_feature(mel_array)
                if not valid:
                    raise ValueError(f"Feature validation failed: {msg}")
                
                np.save(cache_path, mel_array)
                processed_count += 1
                status = "processed"
                
            except Exception as e:
                failed_count += 1
                status = "failed"
                failures.append({
                    "file_id": file_id,
                    "source_path": source_path,
                    "partition": partition,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                print(f"Failed {file_id}: {str(e)}")
                
        # Only log success/skipped metadata if it's not failed
        if status in ["processed", "skipped_valid"]:
            cache_entries.append({
                "file_id": file_id,
                "original_file_path": source_path,
                "cache_path": str(cache_path),
                "label": row['label'],
                "label_name": row['label_name'],
                "speaker_id": row['speaker_id'],
                "attack_id": row['attack_id'],
                "partition": partition,
                "feature_shape": "(128, 251)",
                "dtype": "float32",
                "status": status
            })
            
        # Logging
        if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
            elapsed = time.time() - start_time
            rate = (processed_count + skipped_count + failed_count) / elapsed
            rem_files = total_files - (idx + 1)
            rem_time = rem_files / max(rate, 0.001)
            print(f"[{idx+1}/{total_files}] Processed: {processed_count} | Skipped: {skipped_count} | Failed: {failed_count} | Elapsed: {elapsed:.1f}s | Est. Rem: {rem_time:.1f}s")
            
    # Save Metadata
    if cache_entries:
        cache_df = pd.DataFrame(cache_entries)
        # If resume, we might want to merge, but we are generating the partition fully (or up to max-files).
        # We will just write/overwrite it for the files processed in this run. 
        # Actually, if resume, we should merge with existing cache.csv. 
        # To be safe, if resume and file exists, load existing, update/append, then save.
        if args.resume and cache_csv_path.exists():
            old_cache_df = pd.read_csv(cache_csv_path)
            # drop rows that we just processed to avoid duplicates
            old_cache_df = old_cache_df[~old_cache_df['file_id'].isin(cache_df['file_id'])]
            cache_df = pd.concat([old_cache_df, cache_df], ignore_index=True)
            
        cache_df.to_csv(cache_csv_path, index=False)
        
    # Save Failures
    failures_df = pd.DataFrame(failures) if failures else pd.DataFrame(columns=["file_id", "source_path", "partition", "error_type", "error_message"])
    if args.resume and failures_csv_path.exists():
        try:
            old_failures_df = pd.read_csv(failures_csv_path)
            # update logic for failures: just append and drop duplicates
            failures_df = pd.concat([old_failures_df, failures_df], ignore_index=True).drop_duplicates(subset=['file_id'])
        except pd.errors.EmptyDataError:
            pass
            
    failures_df.to_csv(failures_csv_path, index=False)
    
    # Save Build Report
    elapsed = time.time() - start_time
    report = {
        "partition": partition,
        "total_files": total_files,
        "processed_files": processed_count,
        "skipped_files": skipped_count,
        "failed_files": failed_count,
        "elapsed_seconds": elapsed,
        "average_seconds_per_file": elapsed / max((processed_count + skipped_count + failed_count), 1),
        "feature_shape": [128, 251],
        "feature_dtype": "float32"
    }
    
    report_path = METADATA_DIR / f"{partition}_cache_build_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n--- {partition.upper()} CACHE COMPLETE ---")
    print(f"Total: {total_files} | Processed: {processed_count} | Skipped: {skipped_count} | Failed: {failed_count}")

if __name__ == "__main__":
    main()
