import os
import json
import hashlib
import shutil

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def task_1_preserve_results():
    print("Preserving final results...")
    shutil.copyfile(
        "outputs/metrics/final_results.json",
        "outputs/metrics/final_frozen_results_snapshot.json"
    )
    
    # Check hashes for Integrity check (Task 19 start)
    hashes = {
        "checkpoint": compute_md5("models/cnn_real/best_cnn_real.pth"),
        "frozen_configuration": compute_md5("outputs/metrics/final_frozen_configuration.json"),
        "final_results": compute_md5("outputs/metrics/final_results.json"),
        "evaluation_metadata": compute_md5("data/metadata/asvspoof2021_df_evaluation_metadata.csv")
    }
    
    with open("outputs/metrics/initial_integrity_hashes.json", "w") as f:
        json.dump(hashes, f, indent=4)
        
    print("Snapshot and hashes saved.")

if __name__ == "__main__":
    task_1_preserve_results()
