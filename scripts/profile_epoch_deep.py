from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
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
from src.training.ema import ModelEMA
from src.validate import validate as run_validation
from src.utils import ensure_dir, get_device, save_checkpoint, seed_everything


def run_deep_profile():
    print("=" * 80)
    print("ISIC 2024 DEEP EPOCH STAGE PROFILER")
    print("=" * 80)

    config = Config.from_yaml("configs/kaggle_config.yaml")
    config.batch_size = 32
    config.num_epochs = 1
    config.use_fp16 = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    gpu_count = torch.cuda.device_count() if is_cuda else 0
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"

    print(f"Device: {device} | CUDA: {is_cuda} | Active GPUs: {gpu_count} ({device_name})")

    # Stage 1: Metadata Feature Computation
    t0 = time.perf_counter()
    train_df, val_df = get_fold_dataframes(config.train_metadata_path, fold_idx=0, n_splits=5)
    sample_train = train_df.iloc[:640].reset_index(drop=True)
    sample_val = val_df.iloc[:320].reset_index(drop=True)

    sample_train = enrich_metadata(sample_train)
    sample_val = enrich_metadata(sample_val)

    processor = MetadataProcessor()
    train_meta = processor.fit_transform(sample_train)
    val_meta = processor.transform(sample_val)
    t_metadata = time.perf_counter() - t0
    print(f"  [STAGE 1] Metadata Feature Computation : {t_metadata*1000.0:.2f} ms")

    # Stage 2: HDF5 & Dataset Instantiation
    t0 = time.perf_counter()
    train_dataset = ISICDataset(
        sample_train, config.train_image_dir,
        transform=build_transforms(train=True, image_size=224, use_advanced=False),
        target_col=config.target_column, image_id_col=config.image_id_column,
        metadata_tensor=train_meta,
    )
    val_dataset = ISICDataset(
        sample_val, config.train_image_dir,
        transform=build_transforms(train=False, image_size=224),
        target_col=config.target_column, image_id_col=config.image_id_column,
        metadata_tensor=val_meta,
    )
    t_dataset_init = time.perf_counter() - t0
    print(f"  [STAGE 2] Dataset Instantiation         : {t_dataset_init*1000.0:.2f} ms")

    # Measure HDF5 Raw Reading Speed over 50 items
    t0 = time.perf_counter()
    for idx in range(min(50, len(train_dataset))):
        _ = train_dataset[idx]
    t_hdf5_read = time.perf_counter() - t0
    avg_hdf5_ms = (t_hdf5_read / 50.0) * 1000.0
    print(f"  [STAGE 2] HDF5 Reading Latency          : {avg_hdf5_ms:.2f} ms / sample ({1000.0/avg_hdf5_ms:.1f} samples/sec)")

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Model & Optimization Setup
    metadata_dim = train_meta.shape[1]
    model = FusionModel(backbone_name="tf_efficientnetv2_s", metadata_dim=metadata_dim, pretrained=False).to(device)
    raw_model = model.module if hasattr(model, "module") else model
    ema = ModelEMA(raw_model, decay=0.999, device=device)
    criterion = build_loss("asymmetric")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if (config.use_fp16 and is_cuda) else None

    # Granular Timing Accumulators
    stage_times = {
        "data_loading": 0.0,
        "forward_pass": 0.0,
        "backward_pass": 0.0,
        "optimizer_step": 0.0,
        "ema_update": 0.0,
        "validation": 0.0,
        "checkpoint_saving": 0.0,
    }

    # Profile 10 training batches
    n_batches = 10
    model.train()

    t_iter_start = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break

        t_data_ready = time.perf_counter()
        stage_times["data_loading"] += (t_data_ready - t_iter_start)

        images = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        # Forward Pass
        t_fwd_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=config.use_fp16 and is_cuda):
            logits = model(images, metadata)
            loss = criterion(logits, labels)
        if is_cuda:
            torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        stage_times["forward_pass"] += (t_fwd_end - t_fwd_start)

        # Backward Pass
        t_bwd_start = time.perf_counter()
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if is_cuda:
            torch.cuda.synchronize()
        t_bwd_end = time.perf_counter()
        stage_times["backward_pass"] += (t_bwd_end - t_bwd_start)

        # Optimizer Step
        t_opt_start = time.perf_counter()
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if is_cuda:
            torch.cuda.synchronize()
        t_opt_end = time.perf_counter()
        stage_times["optimizer_step"] += (t_opt_end - t_opt_start)

        # EMA Update
        t_ema_start = time.perf_counter()
        ema.update(raw_model)
        if is_cuda:
            torch.cuda.synchronize()
        t_ema_end = time.perf_counter()
        stage_times["ema_update"] += (t_ema_end - t_ema_start)

        t_iter_start = time.perf_counter()

    # Profile Validation Stage
    t_val_start = time.perf_counter()
    _ = run_validation(ema.module, val_loader, criterion=criterion, device=device, use_metadata=True)
    if is_cuda:
        torch.cuda.synchronize()
    stage_times["validation"] = time.perf_counter() - t_val_start

    # Profile Checkpoint Saving Stage
    t_ckpt_start = time.perf_counter()
    ckpt_dir = config.output_dir / "checkpoints"
    ensure_dir(ckpt_dir)
    save_checkpoint({
        "epoch": 1, "fold": 0,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_dir / "profile_test.pt")
    if is_cuda:
        torch.cuda.synchronize()
    stage_times["checkpoint_saving"] = time.perf_counter() - t_ckpt_start

    # Summary Report
    total_measured_time = sum(stage_times.values())
    total_images_trained = n_batches * config.batch_size

    print("\n" + "=" * 80)
    print("STAGE TIME BREAKDOWN & PERCENTAGE REPORT")
    print("=" * 80)
    for stage, t_val in stage_times.items():
        pct = (t_val / max(total_measured_time, 1e-5)) * 100.0
        print(f"  {stage:<22s} : {t_val*1000.0:>8.2f} ms | {pct:>6.2f} %")

    print("-" * 80)
    print(f"  Total Measured Stage Time  : {total_measured_time*1000.0:.2f} ms")
    print(f"  Images Processed in Test   : {total_images_trained} train images")
    print(f"  Training Loop Throughput  : {total_images_trained / max(stage_times['data_loading'] + stage_times['forward_pass'] + stage_times['backward_pass'] + stage_times['optimizer_step'] + stage_times['ema_update'], 1e-5):.2f} images/sec")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_deep_profile()
