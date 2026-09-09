import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from src.config import BATCH_SIZE

class CachedMelSpectrogramDataset(Dataset):
    def __init__(self, cache_csv_path):
        """
        Initializes the dataset by loading the cache metadata CSV.
        """
        self.metadata = pd.read_csv(cache_csv_path).to_dict('records')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        
        # Load precomputed .npy feature
        cache_path = row['cache_path']
        feature_np = np.load(cache_path)
        
        # Convert to torch tensor and add channel dimension [1, 128, 251]
        feature_tensor = torch.tensor(feature_np, dtype=torch.float32).unsqueeze(0)
        
        # Label is expected as float32 for BCEWithLogitsLoss
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return {
            "features": feature_tensor,
            "label": label,
            "path": row['original_file_path'],
            "file_id": row['file_id'],
            "speaker_id": str(row['speaker_id']),
            "attack_id": str(row['attack_id'])
        }

def create_cached_dataloaders(train_csv_path, dev_csv_path, batch_size=BATCH_SIZE):
    """
    Creates and returns the training and development dataloaders using cached datasets.
    """
    train_dataset = CachedMelSpectrogramDataset(train_csv_path)
    dev_dataset = CachedMelSpectrogramDataset(dev_csv_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, dev_loader
