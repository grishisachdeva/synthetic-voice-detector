import os
import glob
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

def validate_audio_files():
    metadata_path = 'data/metadata/asvspoof2021_df_evaluation_metadata.csv'
    df = pd.read_csv(metadata_path)
    
    results = []
    excluded = []
    
    print(f"Validating {len(df)} files...")
    
    # We will just do the first few for a quick test if it works, but we need to do all.
    # To be fast, we'll try librosa.load with duration=0.01 just to see if it decodes at all.
    # Actually librosa.load(sr=16000) reads the whole file. We can just use soundfile.read.
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row['file_id']
        path = row['file_path']
        
        decoder_1_status = 'FAIL'
        decoder_2_status = 'SKIPPED'
        final_status = 'UNDECODABLE_BY_CURRENT_STACK'
        sample_rate = None
        duration = None
        error_type = ''
        error_message = ''
        
        # Primary Decoder: soundfile (via librosa or sf directly)
        try:
            # We try sf.read because it's what librosa uses under the hood, but much faster than resample
            data, sr = sf.read(path)
            decoder_1_status = 'SUCCESS'
            final_status = 'DECODABLE'
            sample_rate = sr
            duration = len(data) / sr
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            
            # Secondary Decoder: torchaudio (or audioread via librosa backend='audioread')
            try:
                # Try librosa with audioread
                y, sr = librosa.load(path, sr=None)
                decoder_2_status = 'SUCCESS'
                final_status = 'DECODABLE'
                sample_rate = sr
                duration = len(y) / sr
            except Exception as e2:
                decoder_2_status = 'FAIL'
                error_type += f" | {type(e2).__name__}"
                error_message += f" | {str(e2)}"
        
        results.append({
            'file_id': file_id,
            'path': path,
            'decoder_1_status': decoder_1_status,
            'decoder_2_status': decoder_2_status,
            'final_status': final_status,
            'sample_rate': sample_rate,
            'duration': duration,
            'error_type': error_type,
            'error_message': error_message
        })
        
        if final_status != 'DECODABLE':
            excluded.append({
                'file_id': file_id,
                'path': path,
                'reason': error_message
            })
            
    # Save outputs
    os.makedirs('outputs/metrics', exist_ok=True)
    report_df = pd.DataFrame(results)
    report_df.to_csv('outputs/metrics/df_decode_validation_report.csv', index=False)
    
    with open('outputs/metrics/df_exclusion_report.txt', 'w') as f:
        f.write(f"OFFICIAL EVALUATION FILES: {len(df)}\n")
        f.write(f"DECODABLE FILES: {len(df) - len(excluded)}\n")
        f.write(f"EXCLUDED FILES: {len(excluded)}\n\n")
        f.write("EXCLUSION REASONS:\n")
        for ex in excluded:
            f.write(f"{ex['file_id']} ({ex['path']}): {ex['reason']}\n")
            
    print(f"\nValidation complete. Decodable: {len(df) - len(excluded)}, Excluded: {len(excluded)}")

if __name__ == "__main__":
    validate_audio_files()
