import sys
from pathlib import Path
from src.cached_dataset import CachedMelSpectrogramDataset, create_cached_dataloaders

def test_small_batch():
    train_csv = Path("data/processed/mel_cache/metadata/training_cache.csv")
    dev_csv = Path("data/processed/mel_cache/metadata/development_cache.csv")
    
    if not train_csv.exists() or not dev_csv.exists():
        print("Cache CSVs not found. Please build the cache first.")
        sys.exit(1)
        
    print("Loading Cached Datasets...")
    train_dataset = CachedMelSpectrogramDataset(train_csv)
    dev_dataset = CachedMelSpectrogramDataset(dev_csv)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    
    if len(train_dataset) == 0:
        print("No samples in training cache.")
        sys.exit(1)
        
    sample = train_dataset[0]
    print(f"Sample Feature Shape: {sample['features'].shape}")
    print(f"Sample Label: {sample['label']}")
    
    print("\nCreating DataLoaders...")
    train_loader, dev_loader = create_cached_dataloaders(train_csv, dev_csv, batch_size=4)
    
    print("Fetching one batch...")
    batch = next(iter(train_loader))
    print(f"Batch Features Shape: {batch['features'].shape}")
    print(f"Batch Labels Shape: {batch['label'].shape}")
    print("Test Complete.")

if __name__ == "__main__":
    test_small_batch()
