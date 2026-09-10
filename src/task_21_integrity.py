import os
import json
import hashlib

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def task_21_integrity():
    with open("outputs/metrics/initial_integrity_hashes.json", "r") as f:
        initial_hashes = json.load(f)
        
    current_hashes = {
        "checkpoint": compute_md5("models/cnn_real/best_cnn_real.pth"),
        "frozen_configuration": compute_md5("outputs/metrics/final_frozen_configuration.json"),
        "final_results": compute_md5("outputs/metrics/final_results.json"),
        "evaluation_metadata": compute_md5("data/metadata/asvspoof2021_df_evaluation_metadata.csv")
    }
    
    passed = True
    report = ["FINAL EXPLAINABILITY INTEGRITY AUDIT", "======================================="]
    for key in initial_hashes:
        if initial_hashes[key] == current_hashes[key]:
            report.append(f"[PASS] {key} is unchanged.")
        else:
            report.append(f"[FAIL] {key} WAS MODIFIED!")
            passed = False
            
    report.append(f"\nFinal Integrity Verdict: {'PASS' if passed else 'FAIL'}")
    
    with open('outputs/metrics/explainability_integrity_audit.txt', 'w') as f:
        f.write('\n'.join(report))
        
    print('\n'.join(report))

if __name__ == "__main__":
    task_21_integrity()
