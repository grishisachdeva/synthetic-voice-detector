import numpy as np
import pandas as pd
import os
import librosa
from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram

def verify_cache():
    metadata_path = "data/processed/mel_cache/metadata/asvspoof2021_df_cache.csv"
    if not os.path.exists(metadata_path):
        print("Cache metadata not found!")
        return
        
    df = pd.read_csv(metadata_path)
    samples = df.sample(n=10, random_state=42)
    
    max_diffs = []
    
    for idx, row in samples.iterrows():
        cache_path = row['cache_path']
        original_path = row['original_path']
        
        cached_mel = np.load(cache_path)
        
        out = preprocess_audio(original_path)
        fresh_mel = extract_mel_spectrogram(out["waveform"])
        
        diff = np.max(np.abs(cached_mel - fresh_mel))
        max_diffs.append(diff)
        
        assert np.allclose(cached_mel, fresh_mel, atol=1e-5), f"Mismatch on {original_path}"
        assert cached_mel.shape == (128, 251)
        assert cached_mel.dtype == np.float32
        assert not np.isnan(cached_mel).any()
        assert not np.isinf(cached_mel).any()
        
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/df_feature_cache_validation.txt', 'w') as f:
        f.write("DF FEATURE CACHE VALIDATION\n")
        f.write("===========================\n")
        f.write("Verified: 10 randomly sampled files against live extraction.\n")
        f.write("Validation checks passed: \n")
        f.write("- np.allclose() match\n")
        f.write("- shape == (128, 251)\n")
        f.write("- dtype == float32\n")
        f.write("- No NaN / Inf values\n\n")
        f.write(f"Maximum absolute difference observed: {max(max_diffs):.8e}\n")
        
    print("Cache validation passed.")

if __name__ == "__main__":
    verify_cache()
