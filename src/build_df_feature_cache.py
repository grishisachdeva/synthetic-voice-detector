import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram

cache_dir = "data/processed/mel_cache/asvspoof2021_df"

def process_file(row):
    file_id = row['file_id']
    path = row['file_path']
    label = row['label']
    label_name = row['label_name']
    attack_id = row['attack_id']
    
    try:
        out = preprocess_audio(path)
        mel = extract_mel_spectrogram(out["waveform"])
        if mel.shape != (128, 251):
            return None
            
        cache_path = os.path.join(cache_dir, f"{file_id}.npy")
        np.save(cache_path, mel)
        
        return {
            'file_id': file_id,
            'cache_path': cache_path,
            'label': label,
            'label_name': label_name,
            'attack_id': attack_id,
            'original_path': path
        }
    except Exception:
        return None

def build_df_cache():
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("data/processed/mel_cache/metadata", exist_ok=True)
    
    # Read official metadata and decoder validation report
    metadata_path = 'data/metadata/asvspoof2021_df_evaluation_metadata.csv'
    validation_path = 'outputs/metrics/df_decode_validation_report.csv'
    
    metadata = pd.read_csv(metadata_path)
    validation = pd.read_csv(validation_path)
    
    # Merge to get the decodable subset
    df = metadata.merge(validation, on='file_id', suffixes=('', '_val'))
    decodable_df = df[df['final_status'] == 'DECODABLE']
    
    print(f"Total Official: {len(metadata)}")
    print(f"Decodable: {len(decodable_df)}")
    
    cache_records = []
    metadata_out_path = "data/processed/mel_cache/metadata/asvspoof2021_df_cache.csv"
    
    # Check for resume
    existing_files = set()
    if os.path.exists(metadata_out_path):
        existing_df = pd.read_csv(metadata_out_path)
        existing_files = set(existing_df['file_id'].values)
        cache_records = existing_df.to_dict('records')
        print(f"Resuming cache generation. {len(existing_files)} already processed.")
        
    rows_to_process = [row for _, row in decodable_df.iterrows() if row['file_id'] not in existing_files]
    
    if rows_to_process:
        print(f"Processing {len(rows_to_process)} files in parallel...")
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(process_file, row): row for row in rows_to_process}
            
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                result = future.result()
                if result:
                    cache_records.append(result)
                    
                if (i + 1) % 5000 == 0:
                    pd.DataFrame(cache_records).to_csv(metadata_out_path, index=False)
                    
    pd.DataFrame(cache_records).to_csv(metadata_out_path, index=False)
    print(f"Cache complete. {len(cache_records)} files cached.")

if __name__ == "__main__":
    build_df_cache()
