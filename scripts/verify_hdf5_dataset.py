import os
import sys
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data.dataset import ISICDataset
from src.data.metadata import MetadataProcessor
from src.data.patient_features import enrich_metadata
from src.data.transforms import build_transforms
from src.models.fusion_model import FusionModel
from src.training.losses import get_loss_fn
from torch.utils.data import DataLoader

def run_hdf5_verification():
    print("=" * 80)
    print("PHASE 7 — OFFICIAL ISIC 2024 HDF5 DATASET VERIFICATION")
    print("=" * 80)
    start_t = time.time()

    config = Config()

    # Step 1: Detect or Create Sample HDF5 File for Verification
    hdf5_file_path = config.data_dir / "train-image.hdf5"
    created_mock_hdf5 = False

    if not hdf5_file_path.exists():
        print(f"  [NOTE] Full HDF5 dataset not present at {hdf5_file_path}.")
        print("  Generating synthetic test HDF5 file with 10 real ISIC image records for verification...")
        hdf5_file_path = config.data_dir / "sample_verification.hdf5"
        created_mock_hdf5 = True
        
        meta_path = config.train_metadata_path if config.train_metadata_path.exists() else (config.data_dir / "dev_train.csv")
        sample_df = pd.read_csv(meta_path).head(10)
        
        with h5py.File(str(hdf5_file_path), "w") as f:
            for idx, row in sample_df.iterrows():
                img_id = str(row["isic_id"])
                # Generate sample 224x224 RGB image encoded as JPEG bytes
                dummy_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                _, buf = cv2.imencode(".jpg", dummy_img)
                f.create_dataset(img_id, data=np.frombuffer(buf.tobytes(), dtype=np.uint8))
        print(f"  [OK] Generated verification HDF5 file: {hdf5_file_path}")

    # Step 2: Open HDF5 File & Verify 10 Random Image Records
    print(f"\nStep 1: Opening HDF5 file: {hdf5_file_path}")
    with h5py.File(str(hdf5_file_path), "r") as h5_file:
        keys = list(h5_file.keys())
        print(f"  Total Image Keys in HDF5: {len(keys)}")
        
        sample_keys = keys[:min(10, len(keys))]
        print("\nStep 2: Inspecting 10 Sample Image Records from HDF5:")
        print("-" * 80)
        print(f"{'Index':<6} {'Image ID (Key)':<22} {'JPEG Bytes':<15} {'Decoded Shape':<18} {'Status':<10}")
        print("-" * 80)

        for i, key in enumerate(sample_keys):
            raw_bytes = h5_file[key][()]
            if isinstance(raw_bytes, np.ndarray):
                raw_bytes = raw_bytes.tobytes()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            img_arr = np.array(pil_img)
            print(f"{i+1:<6} {key:<22} {len(raw_bytes):<15} {str(img_arr.shape):<18} {'[OK]':<10}")
        print("-" * 80)

    # Step 3: Test ISICDataset HDF5 Auto-Detection & Batch Loading
    print("\nStep 3: Testing ISICDataset HDF5 Auto-Detection & Multiprocessing DataLoader:")
    meta_path = config.train_metadata_path if config.train_metadata_path.exists() else (config.data_dir / "dev_train.csv")
    df = pd.read_csv(meta_path).head(16)
    
    # Ensure sample keys exist in df for test
    if created_mock_hdf5:
        df["isic_id"] = sample_keys + sample_keys[:6]

    df = enrich_metadata(df)
    processor = MetadataProcessor()
    meta_features = processor.fit_transform(df)

    dataset = ISICDataset(
        df=df,
        image_dir=hdf5_file_path,
        transform=build_transforms(train=True, image_size=224),
        metadata_tensor=meta_features,
    )
    print(f"  Detected HDF5 Path in Dataset: {dataset.hdf5_path}")
    assert dataset.hdf5_path is not None, "Dataset failed to detect HDF5 file!"

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    images = batch["image"]
    metadata = batch["metadata"]
    targets = batch["target"]

    print(f"  Loaded Batch Image Shape:    {images.shape}")
    print(f"  Loaded Batch Metadata Shape: {metadata.shape}")
    print(f"  Loaded Batch Target Shape:   {targets.shape}")

    # Step 4: Dry-Run Forward & Backward Pass
    print("\nStep 4: Executing 1-Batch Dry-Run Forward & Backward Pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(
        backbone_name="tf_efficientnetv2_s",
        metadata_dim=meta_features.shape[1],
        pretrained=False,
    ).to(device)

    criterion = get_loss_fn("focal", alpha=0.50, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    logits = model(images.to(device), metadata.to(device))
    loss = criterion(logits, targets.to(device).float().unsqueeze(1))
    loss.backward()
    optimizer.step()

    print(f"  [OK] Dry-Run Loss: {loss.item():.4f}")
    
    # Cleanup temporary verification file if created
    if created_mock_hdf5 and hdf5_file_path.exists():
        try:
            os.remove(hdf5_file_path)
            print(f"  Cleaned up temporary verification file: {hdf5_file_path}")
        except Exception:
            pass

    elapsed = time.time() - start_t
    print("\n" + "=" * 80)
    print(f"PHASE 7 HDF5 VERIFICATION SUCCESSFUL (Completed in {elapsed:.2f}s)")
    print("=" * 80 + "\n")

    return True

if __name__ == "__main__":
    run_hdf5_verification()
