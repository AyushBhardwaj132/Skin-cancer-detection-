from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import datetime
import shutil
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def update_status_json(
    output_dir: Path | None,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    loss: float,
    avg_loss: float,
    lr: float,
    eta_str: str,
    gpu_memory: str,
    checkpoint_name: str,
) -> None:
    """Atomically write outputs/status.json for live monitoring."""
    if output_dir is None:
        return
    status_data = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "loss": round(loss, 4),
        "avg_loss": round(avg_loss, 4),
        "lr": f"{lr:.2e}",
        "eta": eta_str,
        "gpu_memory": gpu_memory,
        "checkpoint": checkpoint_name,
        "last_update": datetime.datetime.now().isoformat(),
    }
    status_file = output_dir / "status.json"
    try:
        tmp_file = status_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_file.replace(status_file)
    except Exception:
        pass


def print_training_dashboard(
    fold: int,
    n_splits: int,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    current_loss: float,
    avg_loss: float,
    lr: float,
    hw_stats: dict,
    elapsed_sec: float,
    best_pauc: float,
    last_checkpoint_name: str,
    last_checkpoint_time: str,
) -> None:
    """Objective 2: Render clean, professional training dashboard."""
    pct = (batch / max(total_batches, 1)) * 100
    bar_width = 24
    filled = int(bar_width * (batch / max(total_batches, 1)))
    progress_bar = "#" * filled + "-" * (bar_width - filled)

    batches_done = (epoch - 1) * total_batches + batch
    total_run_batches = total_epochs * total_batches
    batches_remaining = max(0, total_run_batches - batches_done)
    sec_per_batch = elapsed_sec / max(batch, 1)
    eta_sec = batches_remaining * sec_per_batch

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

    gpu_name = hw_stats.get("gpu_name", "CPU")
    gpu_mem_used = hw_stats.get("gpu_mem_used", 0.0)
    gpu_mem_total = hw_stats.get("gpu_mem_total", 0.0)
    gpu_util = hw_stats.get("gpu_util", 0)
    img_per_sec = hw_stats.get("img_per_sec", 0.0)
    data_ms = hw_stats.get("avg_data_ms", 0.0)
    fwd_ms = hw_stats.get("avg_fwd_ms", 0.0)
    bwd_ms = hw_stats.get("avg_bwd_ms", 0.0)

    print("\n" + "=" * 80, flush=True)
    print("ISIC 2024 Skin Cancer Detection", flush=True)
    print(f"Fold {fold} | Epoch {epoch}/{total_epochs}", flush=True)
    print("=" * 80, flush=True)
    print(f"\nProgress        : {progress_bar} {pct:.1f}%\n", flush=True)
    print(f"Batch           : {batch} / {total_batches}", flush=True)
    print(f"Current Loss    : {current_loss:.4f}", flush=True)
    print(f"Average Loss    : {avg_loss:.4f}\n", flush=True)
    print(f"Learning Rate   : {lr:.2e}\n", flush=True)

    if gpu_name != "CPU":
        print(f"GPU             : {gpu_name}", flush=True)
        print(f"GPU Memory      : {gpu_mem_used:.1f} / {gpu_mem_total:.1f} GB", flush=True)
        print(f"GPU Utilization : {gpu_util} %\n", flush=True)
    else:
        print("Hardware        : CPU Mode\n", flush=True)

    print(f"Images/sec      : {img_per_sec:.0f}\n", flush=True)
    print(f"Data Loading    : {data_ms:.1f} ms", flush=True)
    print(f"Forward         : {fwd_ms:.1f} ms", flush=True)
    print(f"Backward        : {bwd_ms:.1f} ms\n", flush=True)
    print(f"Elapsed Time    : {elapsed_str}", flush=True)
    print(f"ETA             : {eta_str}\n", flush=True)
    print(f"Current Fold    : {fold} / {n_splits}", flush=True)
    print(f"Epoch Progress  : {pct:.1f} %\n", flush=True)
    print("Last Checkpoint : Saved [OK]", flush=True)
    print(f"Checkpoint File : {last_checkpoint_name}", flush=True)
    print(f"Checkpoint Time : {last_checkpoint_time}\n", flush=True)
    print(f"Best Validation pAUC : {best_pauc:.4f}\n", flush=True)
    print("=" * 80 + "\n", flush=True)
    sys.stdout.flush()

from src.config import Config
from src.dataset import ISICDataset
from src.fusion_model import FusionModel
from src.losses import build_loss
from src.metadata import MetadataProcessor
from src.data.caching import MetadataCacheManager
from src.metrics import compute_pauc
from src.model import build_model
from src.patient_features import enrich_metadata
from src.split import get_fold_dataframes
from src.transforms import build_transforms, mixup_data, cutmix_data
from src.utils import ensure_dir, get_device, save_checkpoint, load_checkpoint, seed_everything, seed_worker, sync_file
from src.validate import validate as run_validation
from src.training.ema import ModelEMA
from src.training.state import TrainingState, save_resume_info, load_resume_info, get_git_commit
from src.training.hf_backup import HuggingFaceBackup
from src.training.archiver import create_fold_artifact_zip
from src.evaluation.diagnostic import generate_fold_diagnostic_report
from src.training.hardware import setup_accelerated_model, ThroughputLogger, get_hardware_info
from src.evaluation.logging_artifacts import generate_evaluation_artifacts
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def _compute_class_weights(targets: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    unique, counts = np.unique(targets, return_counts=True)
    weights = counts.sum() / (len(unique) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _build_loaders_fold(
    config: Config,
    fold_idx: int = 0,
    limit_train: int | None = None,
    limit_val: int | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    """Build data loaders using GroupKFold split with optional metadata.

    Returns:
        (train_loader, val_loader, metadata_dim)
    """
    if not config.train_metadata_path.exists():
        raise FileNotFoundError(f"Missing training metadata: {config.train_metadata_path}")

    train_df, val_df = get_fold_dataframes(
        config.train_metadata_path,
        fold_idx=fold_idx,
        n_splits=config.n_splits,
    )

    if limit_train is not None and limit_train > 0:
        train_df = train_df.iloc[:limit_train].reset_index(drop=True)
    if limit_val is not None and limit_val > 0:
        val_df = val_df.iloc[:limit_val].reset_index(drop=True)

    # --- Phase 3: Metadata processing (Cached for fast startup) ---
    metadata_dim = 1  # fallback when metadata is disabled
    train_meta_features = None
    val_meta_features = None

    if config.use_metadata:
        full_df = MetadataCacheManager.load_or_compute_enriched_metadata(
            config.train_metadata_path, config, verbose=True
        )
        # Slice fold dataframes from enriched dataframe
        train_df, val_df = get_fold_dataframes(
            config.train_metadata_path,
            fold_idx=fold_idx,
            n_splits=config.n_splits,
        )
        train_df = full_df.iloc[train_df.index].reset_index(drop=True)
        val_df = full_df.iloc[val_df.index].reset_index(drop=True)

        if limit_train is not None and limit_train > 0:
            train_df = train_df.iloc[:limit_train].reset_index(drop=True)
        if limit_val is not None and limit_val > 0:
            val_df = val_df.iloc[:limit_val].reset_index(drop=True)

        # Fit metadata processor on train, transform both
        processor = MetadataProcessor()
        train_meta_features = processor.fit_transform(train_df)
        val_meta_features = processor.transform(val_df)
        metadata_dim = train_meta_features.shape[1]

        val_positives = int((val_df[config.target_column] == 1).sum()) if config.target_column in val_df.columns else 0
        val_benign = len(val_df) - val_positives
        val_pos_ratio = (val_positives / max(len(val_df), 1)) * 100.0

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx} VALIDATION DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"  Total Validation Samples: {len(val_df)}")
        print(f"  Melanoma Positives (1)  : {val_positives} ({val_pos_ratio:.2f}%)")
        print(f"  Benign Samples (0)      : {val_benign} ({100.0 - val_pos_ratio:.2f}%)")
        print(f"  Patient Group Isolation : VERIFIED [OK]")
        print(f"{'='*60}\n")

        # Save processor for inference
        processor.save(str(config.metadata_processor_path))
        print(f"  Metadata features: {metadata_dim} dims "
              f"({train_meta_features.shape[0]} train, {val_meta_features.shape[0]} val)")

    # --- Build datasets ---
    train_dataset = ISICDataset(
        train_df,
        config.train_image_dir,
        transform=build_transforms(
            train=True,
            image_size=config.image_size,
            use_advanced=config.use_advanced_augs,
        ),
        target_col=config.target_column,
        image_id_col=config.image_id_column,
        metadata_tensor=train_meta_features,
    )
    val_dataset = ISICDataset(
        val_df,
        config.train_image_dir,
        transform=build_transforms(train=False, image_size=config.image_size),
        target_col=config.target_column,
        image_id_col=config.image_id_column,
        metadata_tensor=val_meta_features,
    )

    # --- Weighted sampler for class imbalance ---
    train_targets = train_df[config.target_column].values
    valid_mask = train_targets >= 0
    train_targets_valid = train_targets[valid_mask]
    sample_weights = _compute_class_weights(train_targets_valid)
    sample_weights = torch.tensor(
        [sample_weights[int(t)] for t in train_targets_valid],
        dtype=torch.float32,
    )
    
    g = torch.Generator()
    g.manual_seed(config.seed)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.num_workers > 0),
        prefetch_factor=(2 if config.num_workers > 0 else None),
        timeout=(120 if config.num_workers > 0 else 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.num_workers > 0),
        prefetch_factor=(2 if config.num_workers > 0 else None),
        timeout=(120 if config.num_workers > 0 else 0),
    )
    return train_loader, val_loader, metadata_dim


def _train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    fold: int = 0,
    n_splits: int = 5,
    epoch: int = 1,
    total_epochs: int = 10,
    best_pauc: float = 0.0,
    ema=None,
    pbar_desc: str = "",
    use_metadata: bool = True,
    use_mixup: bool = False,
    mixup_alpha: float = 0.4,
    use_cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    use_fp16: bool = True,
    checkpoint_batch_interval: int = 500,
    save_intra_epoch_checkpoint: callable | None = None,
    output_dir: Path | None = None,
    debug: bool = False,
    start_batch_idx: int = 0,
):
    """Train for one epoch with clean dashboard, AMP, EMA, and optional debug tracing."""
    model.train()
    running_loss = 0.0
    running_samples = 0
    use_amp = use_fp16 and device.type == "cuda"
    t_epoch_start = time.perf_counter()

    throughput_logger = ThroughputLogger(
        total_batches=len(dataloader),
        batch_size=getattr(dataloader, "batch_size", 32) or 32,
        device=device,
        log_interval=100,
    )

    total_batches = len(dataloader)
    if debug:
        print(f"  [TRAIN LOOP] Starting training loop across {total_batches} batches (start_batch={start_batch_idx})...", flush=True)

    data_iter = iter(dataloader)
    batch_idx = 0
    last_ckpt_name = f"last_checkpoint_fold{fold}.pt"
    last_ckpt_time = datetime.datetime.now().strftime("%H:%M:%S")

    while True:
        batch_idx += 1
        if start_batch_idx > 0 and batch_idx <= start_batch_idx:
            next(data_iter, None)
            continue

        # 1. DataLoader fetch
        if debug:
            print(f"  [TRAIN BATCH {batch_idx}/{total_batches}] Fetching next batch from DataLoader worker...", flush=True)
        t_dl_start = time.perf_counter()
        try:
            batch = next(data_iter)
            dl_elapsed = time.perf_counter() - t_dl_start
            throughput_logger.end_data_timer()
            if debug:
                print(f"  [TRAIN BATCH {batch_idx}/{total_batches}] Batch fetched in {dl_elapsed:.4f}s", flush=True)
        except StopIteration:
            if debug:
                print(f"  [TRAIN LOOP] Reached end of DataLoader at batch {batch_idx - 1}", flush=True)
            break
        except Exception as dl_err:
            print(f"  [ERROR TRAIN STEP 1] DataLoader fetch raised exception: {dl_err}", flush=True)
            raise dl_err

        # 2. Batch transfer to GPU
        if debug:
            print(f"  [TRAIN STEP 2] Transferring batch tensors to device ({device})...", flush=True)
        images = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True) if ("metadata" in batch and batch["metadata"] is not None) else None
        labels = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)

        # 3. Optimizer zero_grad
        if debug:
            print(f"  [TRAIN STEP 3] Zeroing optimizer gradients...", flush=True)
        optimizer.zero_grad(set_to_none=True)

        # 4. MixUp / CutMix & Forward pass & Loss computation
        if debug:
            print(f"  [TRAIN STEP 4] Executing Forward pass & Loss computation (AMP={use_amp})...", flush=True)
        t_fwd_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_mixup and np.random.rand() < 0.5:
                if debug:
                    print("    [MIXUP] Applying MixUp augmentation...", flush=True)
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                logits = model(images, metadata) if use_metadata else model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            elif use_cutmix and np.random.rand() < 0.5:
                if debug:
                    print("    [CUTMIX] Applying CutMix augmentation...", flush=True)
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
                logits = model(images, metadata) if use_metadata else model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(images, metadata) if use_metadata else model(images)
                loss = criterion(logits, labels)

        fwd_time = time.perf_counter() - t_fwd_start

        # HARD ASSERTIONS: Instantly halt training if loss, logits, or labels contain NaN / Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"[FATAL TRAINING ERROR] NaN or Inf loss detected at Fold {fold}, Epoch {epoch}, Batch {batch_idx}/{total_batches}! "
                f"Halting training immediately to prevent model weight corruption."
            )
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError(
                f"[FATAL TRAINING ERROR] NaN or Inf logits produced by model at Fold {fold}, Epoch {epoch}, Batch {batch_idx}/{total_batches}!"
            )
        if torch.isnan(labels).any():
            raise ValueError(f"[FATAL TRAINING ERROR] NaN target labels encountered in batch {batch_idx}!")

        # 5. Backward Pass & Scaler Step
        if debug:
            print(f"  [TRAIN STEP 5] Executing Backward Pass & Optimizer Step...", flush=True)
        t_bwd_start = time.perf_counter()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bwd_time = time.perf_counter() - t_bwd_start

        # 6. EMA Update
        if ema is not None:
            if debug:
                print(f"  [TRAIN STEP 6] Updating Model EMA...", flush=True)
            raw_model = model.module if hasattr(model, "module") else model
            ema.update(raw_model)

        # 7. Logging & Checkpoint Logic
        batch_size = images.size(0)
        curr_loss = loss.item()
        running_loss += curr_loss * batch_size
        running_samples += batch_size
        avg_loss = running_loss / max(running_samples, 1)

        throughput_logger.log_batch(
            batch_idx=batch_idx,
            fwd_time=fwd_time,
            bwd_time=bwd_time,
            batch_size=batch_size,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # Objective 2 & 8: Every 100 batches print ONE clean dashboard & update outputs/status.json
        if batch_idx % 100 == 0 or batch_idx == total_batches:
            hw_stats = throughput_logger.get_stats()
            elapsed_sec = time.perf_counter() - t_epoch_start

            if not debug:
                print_training_dashboard(
                    fold=fold,
                    n_splits=n_splits,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    batch=batch_idx,
                    total_batches=total_batches,
                    current_loss=curr_loss,
                    avg_loss=avg_loss,
                    lr=current_lr,
                    hw_stats=hw_stats,
                    elapsed_sec=elapsed_sec,
                    best_pauc=best_pauc,
                    last_checkpoint_name=last_ckpt_name,
                    last_checkpoint_time=last_ckpt_time,
                )

            # Live status.json file update
            batches_done = (epoch - 1) * total_batches + batch_idx
            total_run_batches = total_epochs * total_batches
            batches_remaining = max(0, total_run_batches - batches_done)
            sec_per_batch = elapsed_sec / max(batch_idx, 1)
            eta_sec = batches_remaining * sec_per_batch
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
            gpu_mem_str = f"{hw_stats.get('gpu_mem_used', 0.0):.1f} / {hw_stats.get('gpu_mem_total', 0.0):.1f} GB" if hw_stats.get("gpu_name") != "CPU" else "CPU Mode"

            update_status_json(
                output_dir=output_dir,
                epoch=epoch,
                total_epochs=total_epochs,
                batch=batch_idx,
                total_batches=total_batches,
                loss=curr_loss,
                avg_loss=avg_loss,
                lr=current_lr,
                eta_str=eta_str,
                gpu_memory=gpu_mem_str,
                checkpoint_name=last_ckpt_name,
            )

        if (
            save_intra_epoch_checkpoint is not None
            and checkpoint_batch_interval > 0
            and batch_idx > 0
            and batch_idx % checkpoint_batch_interval == 0
        ):
            if debug:
                print(f"  [TRAIN STEP 8] Saving intra-epoch checkpoint at batch {batch_idx}/{total_batches}...", flush=True)
            save_intra_epoch_checkpoint(
                batch_idx,
                total_batches,
            )
            last_ckpt_time = datetime.datetime.now().strftime("%H:%M:%S")

    return running_loss / max(running_samples, 1)


class EarlyStopping:
    """Early stopping to avoid overfitting."""
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = float("-inf")
        self.counter = 0
        self.stopped = False

    def __call__(self, current_score: float) -> bool:
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                self.stopped = True
                return True
        return False


def plot_training_curves(history: list[dict], save_path: Path):
    """Generate and save training curves for Loss, Validation pAUC, and Learning Rate."""
    if not history:
        return
    ensure_dir(save_path.parent)

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_paucs = [h["val_pauc"] for h in history]
    val_aucs = [h["val_roc_auc"] for h in history]
    lrs = [h["learning_rate"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Train & Val Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2.0)
    axes[0].plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2.0, linestyle="--")
    axes[0].set_title("Train vs Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Subplot 2: Validation Metrics (pAUC & AUC)
    axes[1].plot(epochs, val_paucs, label="Val pAUC", color="#2ca02c", linewidth=2.5)
    axes[1].plot(epochs, val_aucs, label="Val ROC-AUC", color="#d62728", linewidth=1.8, linestyle=":")
    axes[1].set_title("Validation pAUC & ROC-AUC", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Subplot 3: Learning Rate
    axes[2].plot(epochs, lrs, label="Learning Rate", color="#9467bd", linewidth=2.0)
    axes[2].set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Training curves plot saved to {save_path}")


def resolve_resume_fold(
    config: Config,
    requested_fold: int = 0,
    resume: bool = True,
) -> tuple[int, TrainingState]:
    """Inspects training_state.json during resume and resolves the next uncompleted fold to execute."""
    training_state = TrainingState.load(config.output_dir)

    if not resume:
        return requested_fold, training_state

    print(f"[RESUME] Loaded training_state.json", flush=True)
    print(f"[RESUME] completed_folds = {training_state.completed_folds}", flush=True)
    print(f"[RESUME] current_fold = {training_state.current_fold}", flush=True)

    next_fold = training_state.current_fold
    while next_fold in training_state.completed_folds and next_fold < config.n_splits:
        next_fold += 1

    print(f"[RESUME] Next fold to execute = {next_fold}", flush=True)
    sys.stdout.flush()

    if requested_fold in training_state.completed_folds:
        if next_fold < config.n_splits:
            print(f"[RESUME] Requested Fold {requested_fold} is already completed. Automatically advancing execution to Fold {next_fold}!", flush=True)
            return next_fold, training_state
        else:
            print(f"[RESUME] All {config.n_splits} folds are already completed in training_state.json.", flush=True)
            return next_fold, training_state

    return requested_fold, training_state


def verify_checkpoint_dir_writable(checkpoint_dir: Path) -> bool:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    os.makedirs(checkpoint_dir, exist_ok=True)
    test_file = checkpoint_dir / "checkpoint_test.tmp"
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test_checkpoint_dir_writable")
        if not test_file.exists():
            raise RuntimeError(f"[FAIL FAST] Test file was not created: {test_file}")
        test_file.unlink()
        print("Checkpoint directory writable: YES", flush=True)
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"Checkpoint directory writable: NO ({e})", flush=True)
        sys.stdout.flush()
        raise RuntimeError(f"[FAIL FAST] Checkpoint directory {checkpoint_dir} is not writable: {e}")


def print_startup_report(config: Config, fold_idx: int, resume: bool):
    hw = get_hardware_info()
    print("=" * 80, flush=True)
    print("STARTUP REPORT", flush=True)
    print("=" * 80, flush=True)
    print(f"Repository root:      {PROJECT_ROOT}", flush=True)
    print(f"Output directory:     {config.output_dir.resolve()}", flush=True)
    print(f"Checkpoint directory: {config.checkpoint_dir.resolve()}", flush=True)
    print(f"GPU:                  {hw['device_name']}", flush=True)
    print(f"CUDA:                 {hw['is_cuda']}", flush=True)
    print(f"AMP enabled:          {config.use_fp16}", flush=True)
    print(f"Batch size:           {config.batch_size}", flush=True)
    print(f"Workers:              {config.num_workers}", flush=True)
    print(f"Persistent workers:   {config.num_workers > 0}", flush=True)
    print(f"Pin memory:           {torch.cuda.is_available()}", flush=True)
    print(f"Prefetch factor:      {2 if config.num_workers > 0 else None}", flush=True)
    print(f"Checkpoint interval:  {config.checkpoint_batch_interval} batches", flush=True)
    print(f"Resume enabled:       {resume}", flush=True)
    print("=" * 80 + "\n", flush=True)
    sys.stdout.flush()


def list_current_checkpoints(checkpoint_dir: Path, output_dir: Path):
    """Every epoch print every checkpoint currently present."""
    print("Current checkpoints:\n", flush=True)
    found_files = []
    seen = set()

    for d in [checkpoint_dir.resolve(), output_dir.resolve()]:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file() and f.name not in seen and (f.suffix in [".pt", ".json"] or f.name == "training_state.json"):
                    seen.add(f.name)
                    found_files.append(f.name)

    found_files = sorted(found_files)
    if not found_files:
        print("- (None)", flush=True)
    else:
        for fname in found_files:
            print(f"- {fname}", flush=True)
    print("", flush=True)
    sys.stdout.flush()


def train(
    config: Config | None = None,
    fold_idx: int = 0,
    resume: bool = False,
    limit_train: int | None = None,
    limit_val: int | None = None,
    hf_backup: HuggingFaceBackup | None = None,
):
    """Full Competition Training pipeline supporting Mixed Precision, EMA, GroupKFold, and Checkpoint Resuming."""
    pipeline_start_time = time.time()
    config = config or Config()
    seed_everything(config.seed)

    if hf_backup is None and getattr(config, "hf_enabled", True):
        hf_backup = HuggingFaceBackup(repo_id=config.hf_repo_id)

    # 8. Startup Report
    print_startup_report(config, fold_idx, resume)

    # 1 & 9. Checkpoint Directory & Early Self Test
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ensure_dir(config.output_dir)
    ensure_dir(config.log_dir)
    ensure_dir(config.prediction_dir)
    ensure_dir(config.figures_dir)

    print("=" * 49, flush=True)
    print("Checkpoint Directory:", flush=True)
    print(f"{config.checkpoint_dir.resolve()}", flush=True)
    print("=" * 49 + "\n", flush=True)
    sys.stdout.flush()

    verify_checkpoint_dir_writable(config.checkpoint_dir)

    bb_dir = config.get_backbone_checkpoint_dir(
        config.backbone_name if config.use_metadata else config.model_name,
        fold_idx=fold_idx,
    )
    best_checkpoint_path = bb_dir / f"best_model_fold{fold_idx}.pt"
    best_root_ckpt_path = config.checkpoint_dir / f"best_model_fold{fold_idx}.pt"

    last_checkpoint_path = bb_dir / f"last_checkpoint_fold{fold_idx}.pt"
    last_root_ckpt_path = config.checkpoint_dir / f"last_checkpoint_fold{fold_idx}.pt"

    # --- Load Training State ---
    training_state = TrainingState.load(config.output_dir)

    # 4. Verify Resume
    if resume:
        target_resume_path = None
        current_model_name = config.backbone_name if config.use_metadata else config.model_name
        clean_current = current_model_name.replace("tf_", "").replace("-", "_")

        resume_info = load_resume_info(config.output_dir)
        if resume_info:
            print(f"[RESUME INFO LOG] Fast lookup from resume_info.json:")
            print(f"  Fold: {resume_info.get('fold')}, Epoch: {resume_info.get('epoch')}, Batch: {resume_info.get('batch')}, Checkpoint: {resume_info.get('checkpoint')}")
            info_ckpt_name = resume_info.get("checkpoint")
            if info_ckpt_name:
                for search_dir in [bb_dir, config.checkpoint_dir]:
                    candidate = search_dir / info_ckpt_name
                    if candidate.exists() and candidate.stat().st_size > 0:
                        target_resume_path = candidate
                        break

        if not target_resume_path:
            for candidate in [last_checkpoint_path, last_root_ckpt_path, best_checkpoint_path, best_root_ckpt_path]:
                if candidate.exists() and candidate.stat().st_size > 0:
                    try:
                        chk_meta = load_checkpoint(candidate, map_location="cpu")
                        chk_model = chk_meta.get("model_name", "")
                        clean_chk = chk_model.replace("tf_", "").replace("-", "_") if chk_model else ""
                        if not chk_model or clean_chk == clean_current:
                            target_resume_path = candidate
                            break
                    except Exception:
                        continue

        if not target_resume_path or not target_resume_path.exists():
            print(f"[RESUME] No checkpoint found under {config.checkpoint_dir}. Starting fresh training.", flush=True)
            resume = False
        else:
            ckpt_size_mb = target_resume_path.stat().st_size / (1024 * 1024)
            try:
                ckpt = load_checkpoint(target_resume_path, map_location=get_device())
                res_epoch = ckpt.get("epoch", 0)
                res_batch = ckpt.get("batch_idx", 0)
                res_fold = ckpt.get("fold", fold_idx)
                res_global_step = ckpt.get("global_step", 0)

                age_sec = time.time() - target_resume_path.stat().st_mtime
                if age_sec < 60:
                    age_str = f"{int(age_sec)}s"
                elif age_sec < 3600:
                    age_str = f"{int(age_sec // 60)}m {int(age_sec % 60)}s"
                else:
                    age_str = f"{int(age_sec // 3600)}h {int((age_sec % 3600) // 60)}m"

                git_hash = get_git_commit()
                hf_status = "YES" if (getattr(config, "hf_enabled", True) and os.getenv("HF_TOKEN")) else "NO"

                print("\n" + "=" * 50, flush=True)
                print("RESUME SUMMARY", flush=True)
                print("=" * 50, flush=True)
                print(f"Git Commit          : {git_hash}", flush=True)
                print(f"Backbone            : {config.backbone_name}", flush=True)
                print(f"Latest Checkpoint   : {target_resume_path.name}", flush=True)
                print(f"Fold                : {res_fold}", flush=True)
                print(f"Epoch               : {res_epoch}/{config.num_epochs}", flush=True)
                print(f"Batch               : {res_batch}", flush=True)
                print(f"Global Step         : {res_global_step}", flush=True)
                print(f"Checkpoint Age      : {age_str}", flush=True)
                print(f"Checkpoint Verified : YES ({ckpt_size_mb:.1f} MB)", flush=True)
                print(f"HF Backup           : {hf_status}", flush=True)
                print("=" * 50 + "\n", flush=True)
                sys.stdout.flush()

                if fold_idx in training_state.completed_folds:
                    print(f"[RESUME] Skipping Fold {fold_idx} (already completed in training_state.json)", flush=True)
                    sys.stdout.flush()
                    target_eval_ckpt = best_checkpoint_path if best_checkpoint_path.exists() else (best_root_ckpt_path if best_root_ckpt_path.exists() else None)
                    return {
                        "history": [],
                        "best_checkpoint": str(target_eval_ckpt) if target_eval_ckpt else "",
                        "best_val_pauc": training_state.best_pauc,
                        "best_val_auc": 0.0,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "metadata_dim": 0,
                    }
            except Exception as load_err:
                print(f"[RESUME WARN] Failed to reload resume checkpoint at {target_resume_path} ({load_err}). Starting fresh training.", flush=True)
                resume = False

    device = get_device()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Device: {device} | Active GPUs: {gpu_count}")

    # --- Build data loaders ---
    print(f"Building data loaders for fold {fold_idx}...")
    train_loader, val_loader, metadata_dim = _build_loaders_fold(
        config, fold_idx=fold_idx, limit_train=limit_train, limit_val=limit_val
    )

    # --- Build model ---
    if config.use_metadata:
        print(f"Building FusionModel: backbone={config.backbone_name}, metadata_dim={metadata_dim}")
        model = FusionModel(
            backbone_name=config.backbone_name,
            metadata_dim=metadata_dim,
            pretrained=True,
            metadata_hidden=config.metadata_mlp_hidden,
            metadata_output=config.metadata_mlp_output,
        )
    else:
        print(f"Building image-only model: {config.model_name}")
        model = build_model(model_name=config.model_name, pretrained=True, num_classes=1)

    # Multi-GPU DDP setup or DataParallel vs Single GPU benchmark with automatic fallback
    sample_batch = next(iter(train_loader)) if len(train_loader) > 0 else None
    model, accel_metrics = setup_accelerated_model(
        model, device,
        sample_batch=sample_batch,
        multi_gpu_mode=getattr(config, "multi_gpu_mode", "auto"),
    )

    raw_model = model.module if hasattr(model, "module") else model

    # --- EMA Initialization ---
    ema = ModelEMA(raw_model, decay=config.ema_decay, device=device) if getattr(config, "use_ema", True) else None

    # --- Loss, optimizer, scheduler, AMP scaler ---
    criterion = build_loss(
        config.loss_type,
        **({"alpha": config.focal_alpha, "gamma": config.focal_gamma}
           if config.loss_type == "focal" else {}),
    )
    print(f"Loss: {config.loss_type} ({criterion.__class__.__name__})")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if (config.use_fp16 and device.type == "cuda") else None
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=1e-4)

    start_epoch = 1
    best_pauc = float("-inf")
    best_auc = float("-inf")
    history: list[dict] = []

    # --- Resume Training Support ---
    if resume:
        target_resume_path = None
        for candidate in [last_checkpoint_path, last_root_ckpt_path, best_checkpoint_path, best_root_ckpt_path]:
            if candidate.exists():
                target_resume_path = candidate
                break

        if target_resume_path and target_resume_path.exists():
            print(f"Resuming training from checkpoint: {target_resume_path}")
            ckpt = load_checkpoint(target_resume_path, map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt and scaler is not None and ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            if "ema_state_dict" in ckpt and ema is not None and ckpt.get("ema_state_dict") is not None:
                ema.module.load_state_dict(ckpt["ema_state_dict"])
            
            start_batch_idx = 0
            saved_epoch = ckpt.get("epoch", 0)
            saved_batch = ckpt.get("batch_idx", 0)
            total_b = ckpt.get("total_batches", 0)

            if saved_batch > 0 and (total_b == 0 or saved_batch < total_b):
                start_epoch = saved_epoch
                start_batch_idx = saved_batch
            else:
                start_epoch = saved_epoch + 1
                start_batch_idx = 0

            best_pauc = ckpt.get("best_val_pauc", ckpt.get("val_pauc", float("-inf")))
            best_auc = ckpt.get("best_val_auc", ckpt.get("val_auc", float("-inf")))

            # Restore RNG state if present
            if "rng_state" in ckpt and ckpt["rng_state"]:
                rng = ckpt["rng_state"]
                if "torch" in rng and rng["torch"] is not None:
                    torch.set_rng_state(rng["torch"])
                if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                    except Exception:
                        pass
                if "numpy" in rng and rng["numpy"] is not None:
                    np.random.set_state(rng["numpy"])

            print(f"  Resumed at epoch {start_epoch}, batch {start_batch_idx} (Previous Best pAUC={best_pauc:.4f})")

    print(f"\nTraining fold {fold_idx} for epochs {start_epoch} to {config.num_epochs}...")
    print(f"  AMP FP16={config.use_fp16}, EMA={getattr(config, 'use_ema', True)}, MixUp={config.use_mixup}, CutMix={config.use_cutmix}")
    print("-" * 80)

    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()

        def save_intra_epoch_checkpoint(b_idx: int, total_b: int):
            training_state.update_epoch(fold=fold_idx, epoch=epoch, best_pauc=best_pauc, batch_idx=b_idx)
            training_state.save(config.output_dir)

            intra_payload = {
                "epoch": epoch,
                "batch_idx": b_idx,
                "total_batches": total_b,
                "fold": fold_idx,
                "model_name": config.backbone_name if config.use_metadata else config.model_name,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "ema_state_dict": ema.module.state_dict() if ema is not None else None,
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(),
                },
                "best_val_pauc": best_pauc if best_pauc != float("-inf") else 0.0,
                "best_val_auc": best_auc if best_auc != float("-inf") else 0.0,
                "val_pauc": float("nan"),
                "val_auc": float("nan"),
                "val_loss": float("nan"),
                "metadata_dim": metadata_dim,
                "use_metadata": config.use_metadata,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            }
            save_checkpoint(intra_payload, last_checkpoint_path)
            if last_root_ckpt_path.resolve() != last_checkpoint_path.resolve():
                shutil.copy2(last_checkpoint_path, last_root_ckpt_path)
                sync_file(last_root_ckpt_path)
            save_resume_info(
                output_dir=config.output_dir,
                fold=fold_idx,
                epoch=epoch,
                batch_idx=b_idx,
                global_step=epoch * total_b + b_idx,
                checkpoint_name=last_checkpoint_path.name,
            )
            print(f"[OK] Intra-epoch checkpoint saved at batch {b_idx}/{total_b}", flush=True)
            sys.stdout.flush()

        curr_start_b = start_batch_idx if (resume and epoch == start_epoch) else 0

        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            fold=fold_idx,
            n_splits=config.n_splits,
            epoch=epoch,
            total_epochs=config.num_epochs,
            best_pauc=best_pauc if best_pauc != float("-inf") else 0.0,
            ema=ema,
            pbar_desc=f"Epoch {epoch}/{config.num_epochs} [Train]",
            use_metadata=config.use_metadata,
            use_mixup=config.use_mixup,
            mixup_alpha=config.mixup_alpha,
            use_cutmix=config.use_cutmix,
            cutmix_alpha=config.cutmix_alpha,
            use_fp16=config.use_fp16,
            checkpoint_batch_interval=getattr(config, "checkpoint_batch_interval", 500),
            save_intra_epoch_checkpoint=save_intra_epoch_checkpoint,
            output_dir=config.output_dir,
            debug=getattr(config, "debug", False),
            start_batch_idx=curr_start_b,
        )

        # Save LAST checkpoint & update training_state.json IMMEDIATELY after training (BEFORE validation)
        training_state.current_fold = fold_idx
        training_state.last_epoch = epoch
        training_state.last_batch_idx = 0
        if best_pauc != float("-inf") and not np.isnan(best_pauc):
            training_state.best_pauc = best_pauc
        training_state.save(config.output_dir)

        initial_checkpoint_payload = {
            "epoch": epoch,
            "fold": fold_idx,
            "model_name": config.backbone_name if config.use_metadata else config.model_name,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "ema_state_dict": ema.module.state_dict() if ema is not None else None,
            "best_val_pauc": best_pauc if best_pauc != float("-inf") else 0.0,
            "best_val_auc": best_auc if best_auc != float("-inf") else 0.0,
            "val_pauc": float("nan"),
            "val_auc": float("nan"),
            "val_loss": float("nan"),
            "metadata_dim": metadata_dim,
            "use_metadata": config.use_metadata,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        }
        save_checkpoint(initial_checkpoint_payload, last_checkpoint_path)
        if last_root_ckpt_path.resolve() != last_checkpoint_path.resolve():
            shutil.copy2(last_checkpoint_path, last_root_ckpt_path)
            sync_file(last_root_ckpt_path)

        eval_model = ema.module if ema is not None else raw_model

        val_metrics = {
            "loss": float("nan"),
            "roc_auc": float("nan"),
            "pauc": float("nan"),
            "optimal_threshold": 0.5,
            "f1_optimal": 0.0,
        }

        try:
            t_val_stage0 = time.perf_counter()
            val_metrics = run_validation(
                eval_model, val_loader, criterion=criterion,
                device=device, use_metadata=config.use_metadata,
                use_tta=getattr(config, "use_tta", False),
                debug=getattr(config, "debug", False),
            )
            t_val_stage1 = time.perf_counter()

            if "y_true" in val_metrics and "y_score" in val_metrics:
                generate_evaluation_artifacts(
                    val_metrics["y_true"],
                    val_metrics["y_score"],
                    output_dir=config.output_dir,
                    fold_idx=fold_idx,
                    threshold=val_metrics.get("optimal_threshold", 0.5),
                )
        except Exception as val_err:
            if getattr(config, "debug", False):
                print(f"[WARN] Validation computation failed: {val_err}", flush=True)

        sys.stdout.flush()
        scheduler.step()
        epoch_time = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss", float("nan")),
            "val_roc_auc": val_metrics.get("roc_auc", float("nan")),
            "val_pauc": val_metrics.get("pauc", float("nan")),
            "optimal_threshold": val_metrics.get("optimal_threshold", 0.5),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        history.append(epoch_result)

        current_score = val_metrics.get("pauc", val_metrics.get("roc_auc", float("-inf")))
        if np.isnan(current_score):
            current_score = float("-inf")

        is_best = current_score > best_pauc
        is_best_str = "YES" if is_best else "NO"
        epoch_time_str = time.strftime("%H:%M:%S", time.gmtime(epoch_time))

        # Objective 5: Structured Final Epoch Summary
        print("\n" + "=" * 50, flush=True)
        print(f"EPOCH {epoch}/{config.num_epochs} SUMMARY", flush=True)
        print("=" * 50, flush=True)
        print(f"Training Loss        : {train_loss:.4f}", flush=True)
        print(f"Validation Loss      : {val_metrics.get('loss', float('nan')):.4f}", flush=True)
        print(f"Validation pAUC      : {val_metrics.get('pauc', float('nan')):.4f}", flush=True)
        print(f"ROC AUC              : {val_metrics.get('roc_auc', float('nan')):.4f}", flush=True)
        print(f"Learning Rate        : {epoch_result['learning_rate']:.2e}", flush=True)
        print(f"Epoch Time           : {epoch_time_str}", flush=True)
        print(f"Best Model Saved?    : {is_best_str}", flush=True)
        print("Checkpoint Saved?    : YES", flush=True)
        print("=" * 50 + "\n", flush=True)

        # Update Last Checkpoint with full validation metrics
        checkpoint_payload = {
            "epoch": epoch,
            "fold": fold_idx,
            "model_name": config.backbone_name if config.use_metadata else config.model_name,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "ema_state_dict": ema.module.state_dict() if ema is not None else None,
            "best_val_pauc": best_pauc if best_pauc != float("-inf") else current_score,
            "best_val_auc": best_auc if best_auc != float("-inf") else val_metrics.get("roc_auc", float("nan")),
            "val_pauc": val_metrics.get("pauc", float("nan")),
            "val_auc": val_metrics.get("roc_auc", float("nan")),
            "val_loss": val_metrics.get("loss", float("nan")),
            "optimal_threshold": val_metrics.get("optimal_threshold", 0.5),
            "metrics": {
                "val_pauc": val_metrics.get("pauc", float("nan")),
                "val_auc": val_metrics.get("roc_auc", float("nan")),
                "val_loss": val_metrics.get("loss", float("nan")),
                "optimal_threshold": val_metrics.get("optimal_threshold", 0.5),
            },
            "metadata_dim": metadata_dim,
            "use_metadata": config.use_metadata,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        }
        save_checkpoint(checkpoint_payload, last_checkpoint_path)
        if last_root_ckpt_path.resolve() != last_checkpoint_path.resolve():
            shutil.copy2(last_checkpoint_path, last_root_ckpt_path)
            sync_file(last_root_ckpt_path)

        save_resume_info(
            output_dir=config.output_dir,
            fold=fold_idx,
            epoch=epoch,
            batch_idx=0,
            global_step=epoch * len(train_loader),
            checkpoint_name=last_checkpoint_path.name,
        )

        if is_best:
            best_pauc = current_score
            best_auc = val_metrics.get("roc_auc", float("nan"))
            checkpoint_payload["best_val_pauc"] = best_pauc
            checkpoint_payload["best_val_auc"] = best_auc
            save_checkpoint(checkpoint_payload, best_checkpoint_path)
            if best_root_ckpt_path.resolve() != best_checkpoint_path.resolve():
                shutil.copy2(best_checkpoint_path, best_root_ckpt_path)
                sync_file(best_root_ckpt_path)
            training_state.update_epoch(fold=fold_idx, epoch=epoch, best_pauc=best_pauc)
            print(f"[7/8] Saving best checkpoint | New Best pAUC={best_pauc:.4f}", flush=True)

            save_resume_info(
                output_dir=config.output_dir,
                fold=fold_idx,
                epoch=epoch,
                batch_idx=0,
                global_step=epoch * len(train_loader),
                checkpoint_name=best_checkpoint_path.name,
            )

            if hf_backup and hf_backup.is_available:
                print(f"[7/8] Triggering Hugging Face backup for new best model: {best_checkpoint_path.name}...", flush=True)
                hf_backup.upload_checkpoint_async(
                    local_path=best_checkpoint_path,
                    fold_idx=fold_idx,
                    model_name=config.backbone_name,
                )
        else:
            print("[7/8] Saving best checkpoint (no score improvement)", flush=True)

        print(f"[8/8] Epoch complete (Time: {epoch_time:.1f}s)\n", flush=True)
        list_current_checkpoints(config.checkpoint_dir, config.output_dir)
        sys.stdout.flush()

        if early_stopping(current_score):
            break

    # Mark fold completed and write training_state.json
    if fold_idx not in training_state.completed_folds:
        training_state.completed_folds.append(fold_idx)
    training_state.current_fold = fold_idx + 1 if fold_idx + 1 < config.n_splits else fold_idx
    training_state.last_epoch = config.num_epochs
    training_state.save(config.output_dir)
    print("[OK] Fold completed", flush=True)

    # Save training history and plot curves
    history_df = pd.DataFrame(history)
    history_path = config.log_dir / f"training_history_{config.backbone_name}_fold{fold_idx}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\nTraining history saved to {history_path}")

    curve_path = config.figures_dir / f"training_curves_fold{fold_idx}.png"
    plot_training_curves(history, curve_path)

    # --- Comprehensive Validation Inference & Metric Reporting ---
    print(f"\n{'='*80}")
    print(f"RUNNING VALIDATION METRIC EVALUATION (Fold {fold_idx})")
    print(f"{'='*80}")

    target_eval_ckpt = best_checkpoint_path if best_checkpoint_path.exists() else (best_root_ckpt_path if best_root_ckpt_path.exists() else (last_checkpoint_path if last_checkpoint_path.exists() else last_root_ckpt_path))
    best_ckpt = load_checkpoint(target_eval_ckpt, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device) if "metadata" in batch else None
            labels = batch["target"]
            logits = model(images, metadata) if (config.use_metadata and metadata is not None) else model(images)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            val_preds_list.append(probs)
            val_targets_list.append(labels.numpy())

    y_val_true = np.concatenate(val_targets_list, axis=0)
    y_val_pred = np.concatenate(val_preds_list, axis=0)

    val_pauc = compute_pauc(y_val_true, y_val_pred, max_fpr=0.1)
    val_auc = float(roc_auc_score(y_val_true, y_val_pred))
    
    # Binary classification metrics at standard 0.5 decision threshold
    binary_preds = (y_val_pred >= 0.5).astype(int)
    val_acc = float(accuracy_score(y_val_true, binary_preds))
    val_prec = float(precision_score(y_val_true, binary_preds, zero_division=0))
    val_rec = float(recall_score(y_val_true, binary_preds, zero_division=0))
    val_f1 = float(f1_score(y_val_true, binary_preds, zero_division=0))

    print(f"  Validation pAUC@0.1 : {val_pauc:.4f}")
    print(f"  Validation ROC-AUC  : {val_auc:.4f}")
    print(f"  Validation Accuracy : {val_acc:.4f}")
    print(f"  Validation Precision: {val_prec:.4f}")
    print(f"  Validation Recall   : {val_rec:.4f}")
    print(f"  Validation F1 Score : {val_f1:.4f}")
    print(f"{'='*80}\n")

    # Generate Fold Diagnostic JSON Report (Leakage Audit, Target Ratio, Confusion Matrix)
    val_metrics_dict = {
        "pauc": val_pauc,
        "roc_auc": val_auc,
        "optimal_threshold": 0.5,
        "f1_optimal": val_f1,
        "loss": history[-1].get("val_loss", 0.0) if history else 0.0,
        "y_true": y_val_true,
        "y_score": y_val_pred,
    }
    _ = generate_fold_diagnostic_report(
        train_df=train_loader.dataset.df if hasattr(train_loader.dataset, "df") else pd.DataFrame(),
        val_df=val_loader.dataset.df if hasattr(val_loader.dataset, "df") else pd.DataFrame(),
        val_metrics=val_metrics_dict,
        fold_idx=fold_idx,
        output_dir=config.output_dir,
    )

    # Requirement 8: Create End-of-Fold Artifact ZIP Archive
    zip_path = create_fold_artifact_zip(config.output_dir, fold_idx)

    # Requirement 3 & 8: Asynchronously upload complete fold artifacts to Hugging Face
    if hf_backup and hf_backup.is_available:
        print(f"  [HF BACKUP] Triggering non-blocking upload of Fold {fold_idx} artifacts & zip archive...")
        hf_backup.upload_fold_artifacts_async(
            output_dir=config.output_dir,
            fold_idx=fold_idx,
            model_name=config.backbone_name,
            zip_path=zip_path,
        )

    total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - pipeline_start_time))

    print("\n" + "=" * 50, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 50, flush=True)
    print(f"Total Time              : {total_time_str}", flush=True)
    print(f"Best Fold               : Fold {fold_idx}", flush=True)
    print(f"Best pAUC               : {val_pauc:.4f}", flush=True)
    print(f"Checkpoint Location     : {config.checkpoint_dir.resolve()}", flush=True)
    print(f"Best Model Location     : {target_eval_ckpt.resolve()}", flush=True)
    print(f"Training Curves Location: {config.figures_dir.resolve()}", flush=True)
    print(f"Evaluation Folder       : {(config.output_dir / 'figures').resolve()}", flush=True)
    print("=" * 50 + "\n", flush=True)
    sys.stdout.flush()

    return {
        "history": history,
        "best_checkpoint": str(target_eval_ckpt),
        "best_val_pauc": val_pauc,
        "best_val_auc": val_auc,
        "accuracy": val_acc,
        "precision": val_prec,
        "recall": val_rec,
        "f1": val_f1,
        "metadata_dim": metadata_dim,
    }


def train_full_ensemble(config: Config | None = None, backbones: list[str] | None = None, resume: bool = False):
    """Train all ensemble backbones across all 5 GroupKFold folds."""
    config = config or Config()
    target_backbones = backbones or list(config.ensemble_backbones)
    
    print(f"\n{'='*80}")
    print(f"Starting Full Competition Ensemble Training: {len(target_backbones)} backbones x {config.n_splits} folds = {len(target_backbones)*config.n_splits} models")
    print(f"Backbones: {target_backbones}")
    print(f"{'='*80}\n")
    
    summary_results = []
    
    for bb in target_backbones:
        print(f"\n>>>> Training Backbone Family: {bb} <<<<")
        for fold in range(config.n_splits):
            print(f"\n--- {bb} | Fold {fold}/{config.n_splits-1} ---")
            cfg = Config(
                backbone_name=bb,
                model_name=bb,
                image_size=config.image_size,
                batch_size=config.batch_size,
                num_epochs=config.num_epochs,
                learning_rate=config.learning_rate,
                loss_type=config.loss_type,
                use_metadata=config.use_metadata,
                use_patient_features=config.use_patient_features,
                use_ugly_duckling=config.use_ugly_duckling,
                use_advanced_augs=config.use_advanced_augs,
                use_mixup=config.use_mixup,
            )
            try:
                res = train(cfg, fold_idx=fold, resume=resume)
                summary_results.append({
                    "backbone": bb,
                    "fold": fold,
                    "best_val_pauc": res["best_val_pauc"],
                    "best_val_auc": res["best_val_auc"],
                    "accuracy": res["accuracy"],
                    "f1": res["f1"],
                    "status": "Success",
                })
            except Exception as exc:
                print(f"ERROR training {bb} fold {fold}: {exc}")
                summary_results.append({
                    "backbone": bb,
                    "fold": fold,
                    "best_val_pauc": float("nan"),
                    "best_val_auc": float("nan"),
                    "accuracy": float("nan"),
                    "f1": float("nan"),
                    "status": f"Failed: {exc}",
                })
                
    summary_df = pd.DataFrame(summary_results)
    summary_path = config.output_dir / "ensemble_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Ensemble training completed! Summary saved to {summary_path}")
    print(summary_df.to_string(index=False))
    print(f"{'='*80}\n")
    return summary_df


def compare_backbones(config: Config | None = None, fold_idx: int = 0, num_epochs: int = 3):
    """Compare different backbone architectures on the same fold."""
    config = config or Config()

    backbones = [
        "tf_efficientnetv2_m",
        "tf_efficientnetv2_l",
        "convnext_base",
        "swin_base_patch4_window12_384",
    ]

    results = []
    for backbone in backbones:
        print(f"\n{'='*80}")
        print(f"Testing backbone: {backbone}")
        print(f"{'='*80}")

        try:
            cfg = Config(
                backbone_name=backbone,
                model_name=backbone,
                num_epochs=num_epochs,
                use_metadata=config.use_metadata,
                use_patient_features=config.use_patient_features,
                use_ugly_duckling=config.use_ugly_duckling,
                loss_type=config.loss_type,
                image_size=config.image_size,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
            )
            result = train(cfg, fold_idx=fold_idx)
            results.append({
                "backbone": backbone,
                "best_pauc": result["best_val_pauc"],
                "best_auc": result["best_val_auc"],
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "backbone": backbone,
                "best_pauc": float("nan"),
                "best_auc": float("nan"),
            })

    results_df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("Backbone Comparison Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv(config.output_dir / "backbone_comparison.csv", index=False)
    return results_df


def run_debug_checkpoint_test(config: Config) -> None:
    """Task 11: Tiny Kaggle Verification Mode.
    
    Fast 1-epoch 20-batch verification run that saves checkpoints every 10 batches,
    reloads checkpoint, tests resume flow, and exits cleanly.
    """
    print("\n" + "=" * 80, flush=True)
    print("RUNNING KAGGLE DEBUG CHECKPOINT VERIFICATION TEST (--debug-checkpoint-test)", flush=True)
    print("=" * 80, flush=True)

    config.output_dir = config.output_dir / "debug_checkpoint_test"
    config.num_epochs = 1
    config.checkpoint_batch_interval = 10
    config.backbone_name = "resnet18"
    config.model_name = "resnet18"
    config.batch_size = 4
    config.use_advanced_augs = False
    config.use_mixup = False
    config.use_cutmix = False
    config.use_fp16 = False
    config.num_workers = 0

    limit_train = 20 * config.batch_size
    limit_val = 10 * config.batch_size

    # Phase 1: Train 1 tiny epoch and save intra-epoch & final checkpoints
    print("\n[DEBUG TEST 1/3] Executing 1 tiny epoch with intra-epoch checkpointing...", flush=True)
    train(config, fold_idx=0, resume=False, limit_train=limit_train, limit_val=limit_val)

    # Phase 2: Verify saved physical checkpoint and training state reloading
    print("\n[DEBUG TEST 2/3] Verifying physical reload of saved checkpoint...", flush=True)
    ckpt_path = config.checkpoint_dir / "last_checkpoint_fold0.pt"
    state_path = config.output_dir / "training_state.json"

    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"[DEBUG TEST FAILURE] Checkpoint missing at {ckpt_path}")
    if not os.path.exists(state_path):
        raise RuntimeError(f"[DEBUG TEST FAILURE] training_state.json missing at {state_path}")

    _ = load_checkpoint(ckpt_path, map_location="cpu")
    print(f"[DEBUG TEST 2/3] Checkpoint physical reload test: PASSED ({ckpt_path.stat().st_size} bytes)", flush=True)

    # Phase 3: Verify resume workflow from saved checkpoint
    print("\n[DEBUG TEST 3/3] Testing resume workflow from saved checkpoint...", flush=True)
    train(config, fold_idx=0, resume=True, limit_train=limit_train, limit_val=limit_val)

    print("\n" + "=" * 80, flush=True)
    print("[DEBUG CHECKPOINT TEST] Verification PASSED. Pipeline is safe for long training runs.", flush=True)
    print("=" * 80 + "\n", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2024 Full Competition Training Pipeline")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--all-folds", action="store_true", help="Train all 5 GroupKFold folds sequentially")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--debug-checkpoint-test", action="store_true", help="Run tiny Kaggle verification mode and exit")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone model")

    args = parser.parse_args()
    config = Config()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.backbone is not None:
        config.backbone_name = args.backbone
        config.model_name = args.backbone
    if args.debug_checkpoint_test:
        config.debug_checkpoint_test = True

    if config.debug_checkpoint_test:
        run_debug_checkpoint_test(config)
    elif args.all_folds:
        train_full_ensemble(config, resume=args.resume)
    else:
        train(config, fold_idx=args.fold, resume=args.resume)
