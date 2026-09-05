# inspect_dataloader.py — Inspect datasets, dataloaders, class balance, speaker overlap, and attack distributions.

import torch
from src.dataset import (
    create_datasets, create_dataloaders,
    get_class_distribution, get_class_weights,
    check_speaker_overlap, get_attack_distribution,
    load_metadata_csv, load_eval_metadata,
)
from src.config import TRAIN_METADATA, DEV_METADATA, EVAL_METADATA


def main():
    print("=" * 60)
    print("     PYTORCH DATASET & DATALOADER INSPECTION")
    print("=" * 60)

    # --- Create datasets ---
    train_ds, dev_ds, eval_ds = create_datasets()
    print(f"\nDataset lengths:")
    print(f"  Training    : {len(train_ds)}")
    print(f"  Development : {len(dev_ds)}")
    print(f"  Evaluation  : {len(eval_ds)}")

    # --- Class distributions ---
    for name, ds in [("Training", train_ds), ("Development", dev_ds), ("Evaluation", eval_ds)]:
        dist = get_class_distribution(ds)
        print(f"\n{name} class distribution:")
        print(f"  BONAFIDE: {dist['bonafide']} ({dist['bonafide_pct']}%)")
        print(f"  SPOOF   : {dist['spoof']} ({dist['spoof_pct']}%)")

    # --- Class weights ---
    weights = get_class_weights(train_ds)
    print(f"\nTraining class weights: {weights.tolist()}")

    # --- Speaker overlap ---
    print("\n--- Speaker Overlap Analysis ---")
    train_meta = load_metadata_csv(TRAIN_METADATA)
    dev_meta = load_metadata_csv(DEV_METADATA)
    eval_meta = load_eval_metadata(EVAL_METADATA)

    check_speaker_overlap(train_meta, dev_meta, "Train", "Dev")
    check_speaker_overlap(train_meta, eval_meta, "Train", "Eval")
    check_speaker_overlap(dev_meta, eval_meta, "Dev", "Eval")

    # --- Attack distributions ---
    print("\n--- Attack/System Distributions ---")
    for name, meta in [("Training", train_meta), ("Development", dev_meta), ("Evaluation", eval_meta)]:
        dist = get_attack_distribution(meta)
        print(f"\n  {name}:")
        for aid, info in dist.items():
            print(f"    {aid}: {info['count']} samples ({info['pct']}%)")

    # --- Single sample from each dataset ---
    print("\n--- Single Sample Inspection ---")
    for name, ds in [("Training", train_ds), ("Development", dev_ds), ("Evaluation", eval_ds)]:
        sample = ds[0]
        print(f"\n  {name} sample[0]:")
        print(f"    features.shape : {sample['features'].shape}")
        print(f"    features.dtype : {sample['features'].dtype}")
        print(f"    label          : {sample['label'].item()}")
        print(f"    path           : {sample['path']}")

    # --- DataLoader batches ---
    print("\n--- DataLoader Batch Inspection ---")
    train_loader, dev_loader, eval_loader = create_dataloaders(train_ds, dev_ds, eval_ds)

    for name, loader in [("Training", train_loader), ("Development", dev_loader), ("Evaluation", eval_loader)]:
        batch = next(iter(loader))
        print(f"\n  {name} batch:")
        print(f"    features.shape : {batch['features'].shape}")
        print(f"    labels.shape   : {batch['label'].shape}")
        print(f"    features.dtype : {batch['features'].dtype}")
        print(f"    labels.dtype   : {batch['label'].dtype}")

    # --- Determinism check (dev) ---
    print("\n--- Determinism Check (Development) ---")
    s1 = dev_ds[0]
    s2 = dev_ds[0]
    match = torch.allclose(s1["features"], s2["features"], atol=1e-6)
    print(f"  dev_ds[0] features match across two calls: {match}")

    print("\n" + "=" * 60)
    print("  Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
