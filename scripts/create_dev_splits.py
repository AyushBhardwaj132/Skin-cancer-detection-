"""
Create Development Mode dataset splits from the full ISIC 2024 training metadata.

Generates:
    data/dev_train.csv     -- balanced training subset
    data/dev_validation.csv -- balanced validation subset

Rules:
    - Patient grouping preserved (no patient leakage)
    - Reproducible via fixed random seed
    - Balanced classes as much as possible
    - Only references existing images (no copying/moving)
    - Original train-metadata.csv is never modified

Usage:
    python scripts/create_dev_splits.py
    python scripts/create_dev_splits.py --train-benign 500 --train-malignant 500 --val-benign 150 --val-malignant 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def _sample_rows_by_patient(
    df: pd.DataFrame,
    patient_ids: list[str],
    target_value: int,
    max_rows: int,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Sample up to max_rows from the given patients, one image per patient first.

    Ensures we spread across patients rather than taking multiple images from the
    same patient.
    """
    # Filter to the target class
    subset = df[(df.patient_id.isin(patient_ids)) & (df.target == target_value)]
    if len(subset) == 0 or max_rows <= 0:
        return pd.DataFrame(columns=df.columns)

    # Take one random image per patient first
    one_per_patient = subset.groupby("patient_id").apply(
        lambda g: g.sample(1, random_state=rng), include_groups=False
    ).reset_index(drop=True)

    if len(one_per_patient) >= max_rows:
        return one_per_patient.sample(max_rows, random_state=rng).reset_index(drop=True)

    # If we need more, add remaining images
    used_idx = set(one_per_patient.index if hasattr(one_per_patient, 'index') else [])
    result = one_per_patient.copy()
    remaining = subset[~subset.index.isin(result.index)]

    if len(result) < max_rows and len(remaining) > 0:
        needed = max_rows - len(result)
        extra = remaining.sample(min(needed, len(remaining)), random_state=rng)
        result = pd.concat([result, extra], ignore_index=True)

    return result.head(max_rows).reset_index(drop=True)


def create_dev_splits(
    metadata_path: Path,
    output_dir: Path,
    train_benign: int = 500,
    train_malignant: int = 500,
    val_benign: int = 150,
    val_malignant: int = 150,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create patient-grouped, class-balanced development splits.

    Algorithm:
    1. Separate patients into malignant-group (patients with >= 1 malignant image)
       and benign-only-group.
    2. Split malignant patients into val and train groups (no overlap).
    3. Split benign-only patients into val and train groups (no overlap).
    4. Sample the requested number of images per class from each group.
    5. Verify zero patient leakage.
    """
    rng = np.random.RandomState(seed)

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df)} rows, {df.patient_id.nunique()} patients")
    print(f"  Benign: {(df.target == 0).sum()}, Malignant: {(df.target == 1).sum()}")

    # Separate patients by whether they have any malignant images
    patient_has_malignant = set(df[df.target == 1].patient_id.unique())
    patient_benign_only = set(df.patient_id.unique()) - patient_has_malignant

    mal_patients = sorted(patient_has_malignant)
    ben_patients = sorted(patient_benign_only)
    rng.shuffle(mal_patients)
    rng.shuffle(ben_patients)

    total_malignant = (df.target == 1).sum()
    print(f"\nMalignant patients: {len(mal_patients)}")
    print(f"Benign-only patients: {len(ben_patients)}")

    # --- Adjust targets if we don't have enough malignant samples ---
    needed_mal = train_malignant + val_malignant
    if needed_mal > total_malignant:
        ratio = train_malignant / needed_mal
        train_malignant = int(total_malignant * ratio)
        val_malignant = total_malignant - train_malignant
        print(f"\n  [NOTE] Adjusted malignant counts to fit available data:")

    print(f"\nTarget split sizes:")
    print(f"  Train: {train_benign} benign + {train_malignant} malignant")
    print(f"  Val:   {val_benign} benign + {val_malignant} malignant")

    # --- Split malignant patients into val and train groups ---
    # Estimate how many patients we need for each set
    # (most malignant patients have 1 image, so patients ~= images)
    val_mal_patient_count = min(val_malignant + 50, len(mal_patients) // 4)
    val_mal_patients = mal_patients[:val_mal_patient_count]
    train_mal_patients = mal_patients[val_mal_patient_count:]

    # --- Split benign-only patients into val and train groups ---
    val_ben_patient_count = min(val_benign + 50, len(ben_patients) // 4)
    val_ben_patients = ben_patients[:val_ben_patient_count]
    train_ben_patients = ben_patients[val_ben_patient_count:]

    # --- Sample images ---
    val_mal_df = _sample_rows_by_patient(df, val_mal_patients, 1, val_malignant, rng)
    train_mal_df = _sample_rows_by_patient(df, train_mal_patients, 1, train_malignant, rng)
    val_ben_df = _sample_rows_by_patient(df, val_ben_patients, 0, val_benign, rng)
    train_ben_df = _sample_rows_by_patient(df, train_ben_patients, 0, train_benign, rng)

    print(f"\nAllocated images:")
    print(f"  Train malignant: {len(train_mal_df)} (from {train_mal_df.patient_id.nunique()} patients)")
    print(f"  Train benign:    {len(train_ben_df)} (from {train_ben_df.patient_id.nunique()} patients)")
    print(f"  Val malignant:   {len(val_mal_df)} (from {val_mal_df.patient_id.nunique()} patients)")
    print(f"  Val benign:      {len(val_ben_df)} (from {val_ben_df.patient_id.nunique()} patients)")

    # --- Combine and shuffle ---
    dev_train = pd.concat([train_mal_df, train_ben_df], ignore_index=True)
    dev_val = pd.concat([val_mal_df, val_ben_df], ignore_index=True)

    dev_train = dev_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    dev_val = dev_val.sample(frac=1, random_state=seed).reset_index(drop=True)

    # --- Verify no patient leakage ---
    train_patients = set(dev_train.patient_id.unique())
    val_patients = set(dev_val.patient_id.unique())
    leaked = train_patients & val_patients

    if leaked:
        raise RuntimeError(f"PATIENT LEAKAGE DETECTED! {len(leaked)} patients appear in both splits: {leaked}")

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "dev_train.csv"
    val_path = output_dir / "dev_validation.csv"

    dev_train.to_csv(train_path, index=False)
    dev_val.to_csv(val_path, index=False)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("DEVELOPMENT SPLITS CREATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\n  Train: {train_path}")
    print(f"    Total: {len(dev_train)}")
    print(f"    Benign: {(dev_train.target == 0).sum()}")
    print(f"    Malignant: {(dev_train.target == 1).sum()}")
    print(f"    Patients: {dev_train.patient_id.nunique()}")
    print(f"\n  Validation: {val_path}")
    print(f"    Total: {len(dev_val)}")
    print(f"    Benign: {(dev_val.target == 0).sum()}")
    print(f"    Malignant: {(dev_val.target == 1).sum()}")
    print(f"    Patients: {dev_val.patient_id.nunique()}")
    print(f"\n  Patient leakage: NONE [OK]")
    print(f"  Random seed: {seed}")
    print(f"{'='*60}")

    return dev_train, dev_val


def main():
    parser = argparse.ArgumentParser(description="Create Development Mode dataset splits")
    parser.add_argument("--metadata", type=str, default="data/train-metadata.csv", help="Path to full training metadata")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for dev CSVs")
    parser.add_argument("--train-benign", type=int, default=500, help="Number of benign training samples")
    parser.add_argument("--train-malignant", type=int, default=500, help="Number of malignant training samples")
    parser.add_argument("--val-benign", type=int, default=150, help="Number of benign validation samples")
    parser.add_argument("--val-malignant", type=int, default=150, help="Number of malignant validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    create_dev_splits(
        metadata_path=Path(args.metadata),
        output_dir=Path(args.output_dir),
        train_benign=args.train_benign,
        train_malignant=args.train_malignant,
        val_benign=args.val_benign,
        val_malignant=args.val_malignant,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
