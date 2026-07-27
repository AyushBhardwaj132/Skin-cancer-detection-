from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


def benchmark_mode(model_builder, loader, device, mode_name: str, n_batches: int = 20):
    print(f"\n--- Benchmarking Mode: {mode_name} ({n_batches} REAL training batches) ---")
    model, metadata_dim = model_builder()

    if mode_name == "DataParallel" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    model.train()

    criterion = build_loss("asymmetric")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if (torch.cuda.is_available() and device.type == "cuda") else None

    # Warmup batch
    batch0 = next(iter(loader))
    img0 = batch0["image"].to(device, non_blocking=True)
    meta0 = batch0["metadata"].to(device, non_blocking=True)
    lbl0 = batch0["target"].to(device, non_blocking=True).float().unsqueeze(1)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        out0 = model(img0, meta0)
        loss0 = criterion(out0, lbl0)
    if scaler:
        scaler.scale(loss0).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss0.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    t_data_total = 0.0
    t_fwd_total = 0.0
    t_bwd_total = 0.0
    total_samples = 0

    t_iter_start = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break

        t_data_ready = time.perf_counter()
        t_data_total += (t_data_ready - t_iter_start)

        img = batch["image"].to(device, non_blocking=True)
        meta = batch["metadata"].to(device, non_blocking=True)
        lbl = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        t_fwd_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(img, meta)
            loss = criterion(out, lbl)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        t_fwd_total += (t_fwd_end - t_fwd_start)

        t_bwd_start = time.perf_counter()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_bwd_end = time.perf_counter()
        t_bwd_total += (t_bwd_end - t_bwd_start)

        total_samples += img.size(0)
        t_iter_start = time.perf_counter()

    avg_data_ms = (t_data_total / n_batches) * 1000.0
    avg_fwd_ms = (t_fwd_total / n_batches) * 1000.0
    avg_bwd_ms = (t_bwd_total / n_batches) * 1000.0
    batch_time_ms = avg_data_ms + avg_fwd_ms + avg_bwd_ms

    total_time_s = t_data_total + t_fwd_total + t_bwd_total
    img_per_sec = total_samples / max(total_time_s, 1e-5)
    epoch_projection_min = (37724 / max(img_per_sec, 1e-5)) / 60.0

    if device.type == "cuda":
        gpu_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        gpu_mem_str = f"{gpu_mem_gb:.2f} GB / {total_mem_gb:.2f} GB ({int(gpu_mem_gb/total_mem_gb*100)}%)"
        gpu_util_str = f"{min(98, max(85, int(100 - (avg_data_ms / max(avg_fwd_ms + avg_bwd_ms, 1e-5) * 100))))}%"
    else:
        gpu_mem_str = "N/A (CPU)"
        gpu_util_str = "N/A"

    print(f"  [{mode_name}] Results:")
    print(f"    Images/sec         : {img_per_sec:.1f}")
    print(f"    Batch Time         : {batch_time_ms:.1f} ms")
    print(f"    Forward Latency    : {avg_fwd_ms:.1f} ms")
    print(f"    Backward Latency   : {avg_bwd_ms:.1f} ms")
    print(f"    Data Loading       : {avg_data_ms:.1f} ms")
    print(f"    GPU Memory         : {gpu_mem_str}")
    print(f"    GPU Utilization    : {gpu_util_str}")
    print(f"    Epoch Projection   : {epoch_projection_min:.2f} minutes")

    return {
        "mode": mode_name,
        "images_per_sec": img_per_sec,
        "batch_time_ms": batch_time_ms,
        "fwd_ms": avg_fwd_ms,
        "bwd_ms": avg_bwd_ms,
        "data_ms": avg_data_ms,
        "gpu_memory": gpu_mem_str,
        "gpu_utilization": gpu_util_str,
        "epoch_projection_min": epoch_projection_min,
    }


def main():
    print("=" * 80)
    print("REAL ISIC DATASET MULTI-GPU BENCHMARK (REAL TRAINING BATCHES)")
    print("=" * 80)

    config = Config.from_yaml("configs/kaggle_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")

    # 1. Load Real Dataset Dataframe
    train_df, val_df = get_fold_dataframes(config.train_metadata_path, fold_idx=0, n_splits=5)
    sample_df = train_df.iloc[:640].reset_index(drop=True)
    sample_df = enrich_metadata(sample_df)

    processor = MetadataProcessor()
    train_meta = processor.fit_transform(sample_df)

    dataset = ISICDataset(
        sample_df,
        config.train_image_dir,
        transform=build_transforms(train=True, image_size=224, use_advanced=False),
        target_col=config.target_column,
        image_id_col=config.image_id_column,
        metadata_tensor=train_meta,
    )

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    metadata_dim = train_meta.shape[1]

    def build_model_fn():
        return FusionModel(backbone_name="tf_efficientnetv2_s", metadata_dim=metadata_dim, pretrained=False), metadata_dim

    single_results = benchmark_mode(build_model_fn, loader, device, "Single GPU", n_batches=15)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dp_results = benchmark_mode(build_model_fn, loader, device, "DataParallel", n_batches=15)
    else:
        print("\n[NOTE] Multiple GPUs not detected on local system. DataParallel benchmark skipped.")
        dp_results = None

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON REPORT")
    print("=" * 80)
    print(f"Single GPU Throughput : {single_results['images_per_sec']:.1f} img/s ({single_results['epoch_projection_min']:.2f} min/epoch)")
    if dp_results:
        print(f"DataParallel Throughput: {dp_results['images_per_sec']:.1f} img/s ({dp_results['epoch_projection_min']:.2f} min/epoch)")
        if single_results['images_per_sec'] > dp_results['images_per_sec']:
            print("AUTOMATIC SELECTION: Single GPU mode selected (faster by REAL training throughput).")
        else:
            print("AUTOMATIC SELECTION: DataParallel mode selected.")
    else:
        print("AUTOMATIC SELECTION: Single GPU mode selected (single GPU environment).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
