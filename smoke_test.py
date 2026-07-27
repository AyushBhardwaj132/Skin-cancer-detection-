"""
Smoke test: 1000 train / 500 val samples, 1 epoch, save checkpoint, exit.
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data.dataset import ISICDataset
from src.data.metadata import MetadataProcessor
from src.data.patient_features import enrich_metadata
from src.data.transforms import build_transforms
from src.data.split import get_fold_dataframes
from src.models.fusion_model import FusionModel
from src.training.losses import get_loss_fn
from src.training.ema import ModelEMA
from src.training.hardware import ThroughputLogger
from src.validate import validate as run_validation
from src.utils import ensure_dir, get_device, save_checkpoint, seed_everything, seed_worker


TRAIN_LIMIT = 1000
VAL_LIMIT = 500


def main():
    # --- Config ---
    config = Config.from_yaml("configs/kaggle_config.yaml")
    config.num_epochs = 1
    config.num_workers = 0  # safe for quick CPU run
    config.use_advanced_augs = False
    config.use_mixup = False
    config.use_cutmix = False
    config.use_fp16 = False
    seed_everything(config.seed)

    device = get_device()
    print(f"Device: {device}")
    print(f"Backbone: {config.backbone_name}")
    print(f"Image size: {config.image_size}")
    print(f"Focal alpha: {config.focal_alpha}")
    print(f"Train limit: {TRAIN_LIMIT}, Val limit: {VAL_LIMIT}")

    # --- Data ---
    train_df, val_df = get_fold_dataframes(
        config.train_metadata_path, fold_idx=0, n_splits=config.n_splits,
    )
    train_df = train_df.iloc[:TRAIN_LIMIT].reset_index(drop=True)
    val_df = val_df.iloc[:VAL_LIMIT].reset_index(drop=True)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # --- Metadata ---
    metadata_dim = 1
    train_meta = None
    val_meta = None
    if config.use_metadata:
        if config.use_patient_features:
            print("Computing patient features...")
            train_df = enrich_metadata(train_df)
            val_df = enrich_metadata(val_df)
        processor = MetadataProcessor()
        train_meta = processor.fit_transform(train_df)
        val_meta = processor.transform(val_df)
        metadata_dim = train_meta.shape[1]
        print(f"Metadata dim: {metadata_dim}")

    # --- Datasets ---
    train_dataset = ISICDataset(
        train_df, config.train_image_dir,
        transform=build_transforms(train=True, image_size=config.image_size, use_advanced=False),
        target_col=config.target_column, image_id_col=config.image_id_column,
        metadata_tensor=train_meta,
    )
    val_dataset = ISICDataset(
        val_df, config.train_image_dir,
        transform=build_transforms(train=False, image_size=config.image_size),
        target_col=config.target_column, image_id_col=config.image_id_column,
        metadata_tensor=val_meta,
    )

    # --- Loaders ---
    g = torch.Generator()
    g.manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)

    # --- Model ---
    print(f"Building FusionModel: backbone={config.backbone_name}, metadata_dim={metadata_dim}")
    model = FusionModel(
        backbone_name=config.backbone_name,
        metadata_dim=metadata_dim, pretrained=True,
        metadata_hidden=config.metadata_mlp_hidden,
        metadata_output=config.metadata_mlp_output,
    ).to(device)

    # --- Loss / Optimizer ---
    criterion = get_loss_fn(config.loss_type, alpha=config.focal_alpha, gamma=config.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)
    print(f"Loss: {config.loss_type} (alpha={config.focal_alpha}, gamma={config.focal_gamma})")

    # --- Train 1 epoch ---
    print("\n--- Training epoch 1/1 ---")
    model.train()
    running_loss = 0.0
    n_samples = 0
    t0 = time.time()

    throughput_logger = ThroughputLogger(
        total_batches=len(train_loader),
        batch_size=config.batch_size,
        device=device,
        log_interval=10,
    )

    for i, batch in enumerate(train_loader, 1):
        throughput_logger.end_data_timer()

        images = batch["image"].to(device)
        metadata = batch["metadata"].to(device) if "metadata" in batch else None
        labels = batch["target"].to(device).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        t_fwd_start = time.perf_counter()
        logits = model(images, metadata) if metadata is not None else model(images)
        loss = criterion(logits, labels)
        fwd_time = time.perf_counter() - t_fwd_start

        t_bwd_start = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bwd_time = time.perf_counter() - t_bwd_start

        bs = images.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        throughput_logger.log_batch(
            batch_idx=i,
            fwd_time=fwd_time,
            bwd_time=bwd_time,
            batch_size=bs,
        )

    train_loss = running_loss / max(n_samples, 1)
    train_time = time.time() - t0
    scheduler.step()
    print(f"Train loss: {train_loss:.4f} | time: {train_time:.1f}s")

    # --- Validate ---
    print("\n--- Validation ---")
    val_metrics = run_validation(model, val_loader, criterion=criterion,
                                 device=device, use_metadata=config.use_metadata)
    print(f"Val loss: {val_metrics['loss']:.4f}")
    print(f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"Val pAUC: {val_metrics.get('pauc', float('nan')):.4f}")

    # --- Save checkpoint (the code path we're verifying) ---
    print("\n--- Saving checkpoint ---")
    ensure_dir(config.checkpoint_dir)
    ckpt_path = config.checkpoint_dir / "smoke_test_fold0.pt"
    checkpoint_payload = {
        "epoch": 1,
        "fold": 0,
        "model_name": config.backbone_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None,
        "ema_state_dict": None,
        "best_val_pauc": val_metrics.get("pauc", float("nan")),
        "best_val_auc": val_metrics["roc_auc"],
        "metadata_dim": metadata_dim,
        "use_metadata": config.use_metadata,
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in asdict(config).items()},
    }
    save_checkpoint(checkpoint_payload, ckpt_path)
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Checkpoint saved: {ckpt_path} ({size_mb:.1f} MB)")

    print("\n[DONE] Smoke test completed successfully.")


if __name__ == "__main__":
    main()
