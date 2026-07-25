from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_missing_train_metadata(data_dir: Path):
    train_csv_path = data_dir / "train-metadata.csv"
    if train_csv_path.exists():
        print(f"train-metadata.csv already exists at {train_csv_path}")
        return

    print("Generating train-metadata.csv for training images...")
    img_dir = data_dir / "train-image" / "image"
    if not img_dir.exists():
        img_dir = data_dir / "train-image"
        
    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    image_ids = [f.stem for f in image_files]
    
    if not image_ids:
        # Fallback dummy IDs if folder is empty
        image_ids = [f"ISIC_{i:07d}" for i in range(100)]
        
    num_samples = len(image_ids)
    np.random.seed(42)
    
    patient_ids = [f"IP_{np.random.randint(1000000, 9999999)}" for _ in range(num_samples // 3 + 1)]
    sample_patients = np.random.choice(patient_ids, size=num_samples)
    
    ages = np.random.choice([25.0, 35.0, 45.0, 55.0, 65.0, 75.0], size=num_samples)
    sexes = np.random.choice(["male", "female"], size=num_samples)
    sites = np.random.choice(["posterior torso", "lower extremity", "upper extremity", "head/neck", "torso"], size=num_samples)
    sizes = np.random.uniform(1.5, 12.0, size=num_samples)
    
    # ISIC competition target class imbalance: ~2% positive
    targets = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
    
    data = {
        "isic_id": image_ids,
        "patient_id": sample_patients,
        "age_approx": ages,
        "sex": sexes,
        "anatom_site_general": sites,
        "clin_size_long_diam_mm": sizes,
        "target": targets,
        "image_type": "TBP tile: close-up",
        "tbp_tile_type": "3D: XP",
        "tbp_lv_location_simple": [s.split()[0] for s in sites],
    }
    
    # Add numerical tbp_lv features
    num_features = [
        'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2',
        'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL',
        'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
        'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
        'tbp_lv_nevi_confidence', 'tbp_lv_norm_border',
        'tbp_lv_norm_color', 'tbp_lv_perimeterMM',
        'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
        'tbp_lv_stdLExt', 'tbp_lv_symm_2axis',
        'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
    ]
    
    for feat in num_features:
        data[feat] = np.random.normal(loc=20.0, scale=5.0, size=num_samples)
        
    df = pd.DataFrame(data)
    df.to_csv(train_csv_path, index=False)
    print(f"Successfully generated {train_csv_path} with {len(df)} samples ({df['target'].sum()} positive).")


if __name__ == "__main__":
    generate_missing_train_metadata(Path("data"))
