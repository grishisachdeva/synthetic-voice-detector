# dataset.py — PyTorch Dataset, DataLoader factory, augmentation, and analysis utilities.

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram
from src.config import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, TARGET_SAMPLES,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    SHUFFLE_TRAIN, SHUFFLE_VALIDATION, SHUFFLE_TEST,
    TRAIN_METADATA, DEV_METADATA, EVAL_METADATA,
    USE_AUGMENTATION, NOISE_PROB, NOISE_SCALE,
    GAIN_PROB, GAIN_RANGE, TIME_SHIFT_PROB, TIME_SHIFT_MAX,
    USE_SPEC_AUGMENTATION, FREQ_MASK_PARAM, TIME_MASK_PARAM,
)

# ============================================================
# Label mapping
# ============================================================

LABEL_MAP = {"bonafide": 0, "spoof": 1}
VALID_LABELS = set(LABEL_MAP.keys())


def _validate_label(label_name: str, file_path: str) -> int:
    """Convert a string label to its numeric value, raising on unknowns."""
    key = label_name.strip().lower()
    if key not in VALID_LABELS:
        raise ValueError(
            f"Invalid label '{label_name}' for file '{file_path}'. "
            f"Expected one of {sorted(VALID_LABELS)}."
        )
    return LABEL_MAP[key]


# ============================================================
# Metadata loading helpers
# ============================================================

def load_metadata_csv(csv_path: str) -> list[dict]:
    """Load a metadata CSV and return a list of row dicts."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_eval_metadata(manifest_path: str) -> list[dict]:
    """Load only the 'eval' partition rows from the unified manifest."""
    rows = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["partition"].strip().lower() == "eval":
                # Remap manifest columns to match training/dev metadata schema
                rows.append({
                    "file_path": row["file_path"],
                    "file_id": row["file_id"],
                    "speaker_id": row.get("speaker_id", ""),
                    "label": row["label"],
                    "label_name": row["label_name"],
                    "attack_id": row.get("attack_id", ""),
                    "dataset_partition": "eval",
                })
    return rows


# ============================================================
# Waveform augmentation (applied BEFORE feature extraction)
# ============================================================

def augment_waveform(audio: np.ndarray) -> np.ndarray:
    """Apply mild stochastic augmentations to a waveform (training only)."""
    # 1. Additive Gaussian noise
    if np.random.rand() < NOISE_PROB:
        noise = np.random.normal(0, NOISE_SCALE, size=audio.shape).astype(np.float32)
        audio = audio + noise

    # 2. Random gain
    if np.random.rand() < GAIN_PROB:
        gain = np.random.uniform(*GAIN_RANGE)
        audio = audio * gain

    # 3. Random circular time shift
    if np.random.rand() < TIME_SHIFT_PROB:
        shift = np.random.randint(-TIME_SHIFT_MAX, TIME_SHIFT_MAX)
        audio = np.roll(audio, shift)

    return audio.astype(np.float32)


# ============================================================
# Spectrogram augmentation (SpecAugment-style masking)
# ============================================================

def augment_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Apply time and frequency masking to a spectrogram (training only)."""
    spec = spec.copy()
    n_mels, n_frames = spec.shape

    # Frequency masking
    f = np.random.randint(0, FREQ_MASK_PARAM + 1)
    f0 = np.random.randint(0, max(n_mels - f, 1))
    spec[f0: f0 + f, :] = 0.0

    # Time masking
    t = np.random.randint(0, TIME_MASK_PARAM + 1)
    t0 = np.random.randint(0, max(n_frames - t, 1))
    spec[:, t0: t0 + t] = 0.0

    return spec


# ============================================================
# PyTorch Dataset
# ============================================================

class AudioDeepfakeDataset(Dataset):
    """
    PyTorch Dataset that converts metadata rows into model-ready tensors.

    Each __getitem__ call:
        1. Loads and preprocesses the audio (preprocess_audio).
        2. Optionally augments the waveform (training only).
        3. Extracts a log-Mel spectrogram.
        4. Optionally augments the spectrogram (training only, if enabled).
        5. Wraps the result in a dict with feature tensor, label, and metadata.

    No data is preloaded into RAM — everything is computed on demand.
    """

    def __init__(self, metadata: list[dict], split: str = "train", training: bool = False):
        """
        Parameters
        ----------
        metadata : list[dict]
            Rows from a metadata CSV. Required keys: file_path, label_name.
        split : str
            One of 'train', 'dev', 'eval'. Used for logging only.
        training : bool
            If True, apply waveform/spectrogram augmentation.
        """
        if not metadata:
            raise ValueError(f"Metadata list is empty for split '{split}'.")

        self.metadata = metadata
        self.split = split
        self.training = training

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata[idx]
        audio_path = row["file_path"]

        # --- Validate label ---
        label_int = _validate_label(row["label_name"], audio_path)

        # --- Preprocess (load → mono → resample → normalize → trim → pad/crop) ---
        result = preprocess_audio(audio_path)
        waveform = result["waveform"]

        # --- Optional waveform augmentation (training only) ---
        if self.training and USE_AUGMENTATION:
            waveform = augment_waveform(waveform)

        # --- Feature extraction (log-Mel spectrogram) ---
        spectrogram = extract_mel_spectrogram(waveform, sample_rate=SAMPLE_RATE)

        # --- Optional spectrogram augmentation (training only) ---
        if self.training and USE_SPEC_AUGMENTATION:
            spectrogram = augment_spectrogram(spectrogram)

        # --- Convert to tensor with channel dim: [1, 128, 251] ---
        features = torch.from_numpy(spectrogram).unsqueeze(0)  # (1, N_MELS, T)
        label = torch.tensor(label_int, dtype=torch.float32)

        sample = {
            "features": features,
            "label": label,
            "path": audio_path,
        }

        # Include optional metadata fields if present
        if "speaker_id" in row:
            sample["speaker_id"] = row["speaker_id"]
        if "attack_id" in row:
            sample["attack_id"] = row["attack_id"]

        return sample


# ============================================================
# Factory functions
# ============================================================

def create_datasets() -> tuple:
    """
    Create train, dev, and eval AudioDeepfakeDataset instances
    from the existing metadata CSVs.

    Returns
    -------
    (train_dataset, dev_dataset, eval_dataset)
    """
    train_meta = load_metadata_csv(TRAIN_METADATA)
    dev_meta = load_metadata_csv(DEV_METADATA)
    eval_meta = load_eval_metadata(EVAL_METADATA)

    train_dataset = AudioDeepfakeDataset(train_meta, split="train", training=True)
    dev_dataset = AudioDeepfakeDataset(dev_meta, split="dev", training=False)
    eval_dataset = AudioDeepfakeDataset(eval_meta, split="eval", training=False)

    return train_dataset, dev_dataset, eval_dataset


def create_dataloaders(
    train_dataset: Dataset,
    dev_dataset: Dataset,
    eval_dataset: Dataset,
) -> tuple:
    """
    Wrap datasets in DataLoaders with the configured batch/worker settings.

    Returns
    -------
    (train_loader, dev_loader, eval_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_VALIDATION,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TEST,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, dev_loader, eval_loader


# ============================================================
# Analysis utilities
# ============================================================

def get_class_distribution(dataset: AudioDeepfakeDataset) -> dict:
    """Return bonafide/spoof counts and percentages for a dataset."""
    bonafide = sum(1 for row in dataset.metadata if row["label_name"].strip().lower() == "bonafide")
    spoof = sum(1 for row in dataset.metadata if row["label_name"].strip().lower() == "spoof")
    total = bonafide + spoof
    return {
        "bonafide": bonafide,
        "spoof": spoof,
        "total": total,
        "bonafide_pct": round(100 * bonafide / total, 2) if total else 0,
        "spoof_pct": round(100 * spoof / total, 2) if total else 0,
    }


def get_class_weights(dataset: AudioDeepfakeDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for binary classification.

    Returns a tensor [w_bonafide, w_spoof] suitable for loss weighting.
    """
    dist = get_class_distribution(dataset)
    total = dist["total"]
    if total == 0 or dist["bonafide"] == 0 or dist["spoof"] == 0:
        return torch.tensor([1.0, 1.0])
    w_bonafide = total / (2 * dist["bonafide"])
    w_spoof = total / (2 * dist["spoof"])
    return torch.tensor([w_bonafide, w_spoof], dtype=torch.float32)


def check_speaker_overlap(meta_a: list[dict], meta_b: list[dict],
                           name_a: str = "A", name_b: str = "B") -> set:
    """
    Check for speaker-ID overlap between two metadata lists.
    Returns the set of overlapping speaker IDs (empty if disjoint).
    """
    spk_a = {row["speaker_id"] for row in meta_a if row.get("speaker_id")}
    spk_b = {row["speaker_id"] for row in meta_b if row.get("speaker_id")}
    overlap = spk_a & spk_b

    print(f"  {name_a} unique speakers: {len(spk_a)}")
    print(f"  {name_b} unique speakers: {len(spk_b)}")
    print(f"  Overlap: {len(overlap)} — {overlap if overlap else '{}'}")

    if overlap:
        print(f"  WARNING: speaker leakage detected between {name_a} and {name_b}!")
    else:
        print(f"  OK: No speaker overlap between {name_a} and {name_b}.")

    return overlap


def get_attack_distribution(metadata: list[dict]) -> dict:
    """Return attack-ID sample counts."""
    counts: dict[str, int] = {}
    for row in metadata:
        aid = row.get("attack_id", "-")
        counts[aid] = counts.get(aid, 0) + 1
    total = sum(counts.values())
    return {aid: {"count": c, "pct": round(100 * c / total, 2)} for aid, c in sorted(counts.items())}
