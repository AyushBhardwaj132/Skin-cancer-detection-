from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data.dataset import ISICDataset
from src.data.split import get_fold_dataframes
from src.data.metadata import MetadataProcessor
from src.data.patient_features import enrich_metadata
from src.data.transforms import build_transforms
from src.models.fusion_model import FusionModel
from src.training.losses import build_loss
from torch.utils.data import DataLoader


def profile_pipeline():
    print("=" * 80)
    print("ISIC 2024 PIPELINE PROFILER & BOTTLENECK DIAGNOSTIC")
    print("=" * 80)

    config = Config.from_yaml("configs/kaggle_config.yaml")
    config.batch_size = 32
    config.use_fp16 = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (CUDA Available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")

    # 1. Load Dataframe & Metadata
    train_df, val_df = get_fold_dataframes(config.train_metadata_path, fold_idx=0, n_splits=5)
    sample_df = train_df.iloc[:200].reset_index(drop=True)
    sample_df = enrich_metadata(sample_df)

    processor = MetadataProcessor()
    train_meta = processor.fit_transform(sample_df)

    # 2. Profile Dataset __getitem__
    dataset = ISICDataset(
        sample_df,
        config.train_image_dir,
        transform=build_transforms(train=True, image_size=224, use_advanced=False),
        target_col=config.target_column,
        image_id_col=config.image_id_column,
        metadata_tensor=train_meta,
    )

    print(f"\n[DATASET] Detected HDF5 Path: {dataset.hdf5_path}")

    # Measure __getitem__ latency over 100 samples
    t_start = time.perf_counter()
    for i in range(min(100, len(dataset))):
        _ = dataset[i]
    t_end = time.perf_counter()
    avg_getitem_ms = ((t_end - t_start) / 100.0) * 1000.0
    print(f"[DATASET] Average __getitem__ latency: {avg_getitem_ms:.2f} ms / sample ({1000.0 / avg_getitem_ms:.1f} samples/sec)")

    # 3. Profile DataLoader Data Loading Speed
    num_workers_list = [0, 2, 4] if sys.platform != "win32" else [0]
    best_loader_throughput = 0.0

    for nw in num_workers_list:
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(nw > 0),
        )

        t_load_start = time.perf_counter()
        count = 0
        for batch in loader:
            count += batch["image"].size(0)
        t_load_end = time.perf_counter()

        throughput = count / (t_load_end - t_load_start)
        print(f"[DATALOADER] num_workers={nw} -> {throughput:.1f} images/second (Total time for {count} samples: {t_load_end - t_load_start:.2f}s)")
        if throughput > best_loader_throughput:
            best_loader_throughput = throughput

    # 4. Profile Model Forward & Backward Pass
    metadata_dim = train_meta.shape[1]
    model = FusionModel(
        backbone_name="tf_efficientnetv2_s",
        metadata_dim=metadata_dim,
        pretrained=False,
    ).to(device)

    criterion = build_loss("asymmetric")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if (config.use_fp16 and device.type == "cuda") else None

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    # Warmup
    model.train()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    metadata = batch["metadata"].to(device)
    labels = batch["target"].to(device).float().unsqueeze(1)

    with torch.amp.autocast("cuda", enabled=config.use_fp16 and device.type == "cuda"):
        logits = model(images, metadata)
        loss = criterion(logits, labels)
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # Time timing Breakdown
    n_batches = 5
    t_data_total = 0.0
    t_fwd_bwd_total = 0.0

    t_iter_start = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        t_data_end = time.perf_counter()
        t_data_total += (t_data_end - t_iter_start)

        images = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        t_fwd_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=config.use_fp16 and device.type == "cuda"):
            logits = model(images, metadata)
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        t_fwd_end = time.perf_counter()
        t_fwd_bwd_total += (t_fwd_end - t_fwd_start)
        t_iter_start = time.perf_counter()

    total_samples = n_batches * config.batch_size
    avg_data_time = t_data_total / n_batches
    avg_fwd_bwd_time = t_fwd_bwd_total / n_batches
    overall_throughput = total_samples / (t_data_total + t_fwd_bwd_total)

    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN & METRICS REPORT")
    print("=" * 80)
    print(f"  DataLoader Wait Time (per batch) : {avg_data_time * 1000.0:.2f} ms")
    print(f"  Forward + Backward Time (batch)   : {avg_fwd_bwd_time * 1000.0:.2f} ms")
    print(f"  Total Images Processed            : {total_samples}")
    print(f"  Overall Training Throughput       : {overall_throughput:.2f} images/second")
    print(f"  Estimated Time per Full Epoch    : {37724 / max(overall_throughput, 1e-5) / 60.0:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    profile_pipeline()
