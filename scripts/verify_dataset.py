"""Quick dataset verification script."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path

data_dir = Path("data")
img_dir = data_dir / "train-image" / "image"
metadata_path = data_dir / "train-metadata.csv"

df = pd.read_csv(metadata_path)
print(f"Total metadata rows: {len(df)}")
print(f"Benign (target=0): {(df.target == 0).sum()}")
print(f"Malignant (target=1): {(df.target == 1).sum()}")
print(f"Unique patients: {df.patient_id.nunique()}")

exts = [".jpg", ".jpeg", ".png", ".webp"]
found = 0
missing = 0
missing_ids = []

for iid in df.isic_id:
    ok = False
    for e in exts:
        if (img_dir / f"{iid}{e}").exists():
            ok = True
            break
    if ok:
        found += 1
    else:
        missing += 1
        if len(missing_ids) < 5:
            missing_ids.append(iid)

print(f"\nImage verification:")
print(f"  Found: {found}")
print(f"  Missing: {missing}")
if missing_ids:
    print(f"  Sample missing IDs: {missing_ids}")

# Check malignant patient distribution
mal_df = df[df.target == 1]
print(f"\nMalignant details:")
print(f"  Total malignant samples: {len(mal_df)}")
print(f"  Unique malignant patients: {mal_df.patient_id.nunique()}")
print(f"  Images per malignant patient (top 5):")
print(mal_df.groupby("patient_id").size().sort_values(ascending=False).head())
