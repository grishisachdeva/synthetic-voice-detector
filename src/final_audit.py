import pandas as pd
import json
import os

def final_leakage_audit():
    print("Performing final leakage audit...")
    df_eval = pd.read_csv('data/processed/mel_cache/metadata/asvspoof2021_df_cache.csv')
    eval_ids = set(df_eval['file_id'].values)
    
    train_ids = set()
    if os.path.exists('data/processed/mel_cache/metadata/training_cache.csv'):
        train_df = pd.read_csv('data/processed/mel_cache/metadata/training_cache.csv')
        train_ids = set(train_df['file_id'].values)
        
    dev_ids = set()
    if os.path.exists('data/processed/mel_cache/metadata/development_cache.csv'):
        dev_df = pd.read_csv('data/processed/mel_cache/metadata/development_cache.csv')
        dev_ids = set(dev_df['file_id'].values)
        
    overlap_train = eval_ids.intersection(train_ids)
    overlap_dev = eval_ids.intersection(dev_ids)
    
    status = "PASS" if len(overlap_train) == 0 and len(overlap_dev) == 0 else "FAIL"
    
    report = [
        "FINAL EVALUATION LEAKAGE AUDIT",
        "==============================",
        f"Overlap with Training Cache: {len(overlap_train)} files",
        f"Overlap with Development Cache: {len(overlap_dev)} files",
        "",
        f"Result: {status}"
    ]
    
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/final_evaluation_leakage_audit.txt', 'w') as f:
        f.write('\n'.join(report))
        
    print('\n'.join(report))

if __name__ == "__main__":
    final_leakage_audit()
