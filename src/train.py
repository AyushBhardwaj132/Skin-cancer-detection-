from __future__ import annotations

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

import shutil
from src.config import Config
from src.dataset import ISICDataset
from src.fusion_model import FusionModel
from src.losses import build_loss
from src.metadata import MetadataProcessor
from src.metrics import compute_pauc
from src.model import build_model
from src.patient_features import enrich_metadata
from src.split import get_fold_dataframes
from src.transforms import build_transforms, mixup_data, cutmix_data
from src.utils import ensure_dir, get_device, save_checkpoint, load_checkpoint, seed_everything, seed_worker
from src.validate import validate as run_validation
from src.training.ema import ModelEMA
from src.training.state import TrainingState
from src.training.hardware import setup_accelerated_model, ThroughputLogger
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

    # --- Phase 3: Metadata processing ---
    metadata_dim = 1  # fallback when metadata is disabled
    train_meta_features = None
    val_meta_features = None

    if config.use_metadata:
        # Enrich with patient-level features and ugly duckling score
        if config.use_patient_features:
            print("  Computing patient features & ugly duckling scores...")
            train_df = enrich_metadata(train_df)
            val_df = enrich_metadata(val_df)

        # Fit metadata processor on train, transform both
        processor = MetadataProcessor()
        train_meta_features = processor.fit_transform(train_df)
        val_meta_features = processor.transform(val_df)
        metadata_dim = train_meta_features.shape[1]

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
    )
    return train_loader, val_loader, metadata_dim


def _train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    ema=None,
    pbar_desc="",
    use_metadata: bool = True,
    use_mixup: bool = False,
    mixup_alpha: float = 0.4,
    use_cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    use_fp16: bool = True,
):
    """Train for one epoch with progress bar, AMP, EMA, and optional MixUp/CutMix."""
    model.train()
    running_loss = 0.0
    running_samples = 0
    use_amp = use_fp16 and device.type == "cuda"

    throughput_logger = ThroughputLogger(
        total_batches=len(dataloader),
        batch_size=getattr(dataloader, "batch_size", 32) or 32,
        device=device,
        log_interval=100,
    )

    pbar = tqdm(dataloader, desc=pbar_desc, leave=False)
    for batch_idx, batch in enumerate(pbar, 1):
        throughput_logger.end_data_timer()

        images = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True) if "metadata" in batch else None
        labels = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        t_fwd_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_mixup and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                logits = model(images, metadata) if use_metadata else model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            elif use_cutmix and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
                logits = model(images, metadata) if use_metadata else model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(images, metadata) if use_metadata else model(images)
                loss = criterion(logits, labels)

        t_fwd_end = time.perf_counter()
        fwd_time = t_fwd_end - t_fwd_start

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

        t_bwd_end = time.perf_counter()
        bwd_time = t_bwd_end - t_bwd_start

        if ema is not None:
            raw_model = model.module if hasattr(model, "module") else model
            ema.update(raw_model)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_samples += batch_size
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        throughput_logger.log_batch(
            batch_idx=batch_idx,
            fwd_time=fwd_time,
            bwd_time=bwd_time,
            batch_size=batch_size,
        )

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


def train(config: Config | None = None, fold_idx: int = 0, resume: bool = False):
    """Full Competition Training pipeline supporting Mixed Precision, EMA, GroupKFold, and Checkpoint Resuming."""
    config = config or Config()
    seed_everything(config.seed)

    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.log_dir)
    ensure_dir(config.prediction_dir)
    ensure_dir(config.figures_dir)

    bb_dir = config.get_backbone_checkpoint_dir(config.backbone_name if config.use_metadata else config.model_name)
    best_checkpoint_path = bb_dir / f"best_model_fold{fold_idx}.pt"
    best_root_ckpt_path = config.checkpoint_dir / f"best_model_fold{fold_idx}.pt"

    last_checkpoint_path = bb_dir / f"last_checkpoint_fold{fold_idx}.pt"
    last_root_ckpt_path = config.checkpoint_dir / f"last_checkpoint_fold{fold_idx}.pt"

    # --- Load Training State ---
    training_state = TrainingState.load(config.output_dir)

    if resume and fold_idx in training_state.completed_folds:
        print(f"Skipping Fold {fold_idx} (already completed in training_state.json)")
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

    device = get_device()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Device: {device} | Active GPUs: {gpu_count}")

    # --- Build data loaders ---
    print(f"Building data loaders for fold {fold_idx}...")
    train_loader, val_loader, metadata_dim = _build_loaders_fold(config, fold_idx=fold_idx)

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
            
            start_epoch = ckpt.get("epoch", 0) + 1
            best_pauc = ckpt.get("best_val_pauc", ckpt.get("val_pauc", float("-inf")))
            best_auc = ckpt.get("best_val_auc", ckpt.get("val_auc", float("-inf")))
            print(f"  Resumed at epoch {start_epoch} (Previous Best pAUC={best_pauc:.4f})")

    print(f"\nTraining fold {fold_idx} for epochs {start_epoch} to {config.num_epochs}...")
    print(f"  AMP FP16={config.use_fp16}, EMA={getattr(config, 'use_ema', True)}, MixUp={config.use_mixup}, CutMix={config.use_cutmix}")
    print("-" * 80)

    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()

        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            ema=ema,
            pbar_desc=f"Epoch {epoch}/{config.num_epochs} [Train]",
            use_metadata=config.use_metadata,
            use_mixup=config.use_mixup,
            mixup_alpha=config.mixup_alpha,
            use_cutmix=config.use_cutmix,
            cutmix_alpha=config.cutmix_alpha,
            use_fp16=config.use_fp16,
        )
        print(f"\n[1/8] Training finished | train_loss={train_loss:.4f}")

        # Save LAST checkpoint & update training_state.json IMMEDIATELY after training (BEFORE validation)
        training_state.current_fold = fold_idx
        training_state.last_epoch = epoch
        if best_pauc != float("-inf") and not np.isnan(best_pauc):
            training_state.best_pauc = best_pauc
        training_state.save(config.output_dir)
        print("✓ Resume information saved")

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
        print("[6/8] Saving last checkpoint")

        print("[2/8] Starting validation")
        eval_model = ema.module if ema is not None else raw_model

        val_metrics = {
            "loss": float("nan"),
            "roc_auc": float("nan"),
            "pauc": float("nan"),
            "optimal_threshold": 0.5,
            "f1_optimal": 0.0,
        }

        try:
            val_metrics = run_validation(
                eval_model, val_loader, criterion=criterion,
                device=device, use_metadata=config.use_metadata,
                use_tta=getattr(config, "use_tta", False),
            )
            print(f"[3/8] Validation finished | val_loss={val_metrics.get('loss', float('nan')):.4f}")

            roc_val = val_metrics.get("roc_auc", float("nan"))
            pauc_val = val_metrics.get("pauc", float("nan"))

            if not np.isnan(roc_val):
                print(f"[4/8] Computing ROC-AUC: {roc_val:.4f}")
            else:
                print("[4/8] Computing ROC-AUC: N/A")

            if not np.isnan(pauc_val):
                print(f"[5/8] Computing pAUC: {pauc_val:.4f}")
            else:
                print("[5/8] Computing pAUC: N/A")

            # Auto-generate evaluation visualization artifacts (ROC, PR, Confusion Matrix)
            if "y_true" in val_metrics and "y_score" in val_metrics:
                generate_evaluation_artifacts(
                    val_metrics["y_true"],
                    val_metrics["y_score"],
                    output_dir=config.output_dir,
                    fold_idx=fold_idx,
                    threshold=val_metrics.get("optimal_threshold", 0.5),
                )
        except Exception as val_err:
            print(f"[WARN] Validation or metric computation failed: {val_err}")
            print(f"[WARN] Continuing pipeline cleanly without freezing.")

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

        print(
            f"Epoch {epoch:>3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.get('loss', float('nan')):.4f} | "
            f"roc_auc={val_metrics.get('roc_auc', float('nan')):.4f} | "
            f"pAUC={val_metrics.get('pauc', float('nan')):.4f} | "
            f"opt_thresh={val_metrics.get('optimal_threshold', 0.5):.2f} | "
            f"lr={epoch_result['learning_rate']:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        current_score = val_metrics.get("pauc", val_metrics.get("roc_auc", float("-inf")))
        if np.isnan(current_score):
            current_score = float("-inf")

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
        print("✓ Checkpoint saved")

        if current_score > best_pauc:
            best_pauc = current_score
            best_auc = val_metrics.get("roc_auc", float("nan"))
            checkpoint_payload["best_val_pauc"] = best_pauc
            checkpoint_payload["best_val_auc"] = best_auc
            save_checkpoint(checkpoint_payload, best_checkpoint_path)
            if best_root_ckpt_path.resolve() != best_checkpoint_path.resolve():
                shutil.copy2(best_checkpoint_path, best_root_ckpt_path)
            print(f"[7/8] Saving best checkpoint | New Best pAUC={best_pauc:.4f}")
            training_state.update_epoch(fold=fold_idx, epoch=epoch, best_pauc=best_pauc)
        else:
            print("[7/8] Saving best checkpoint (no score improvement)")

        print(f"[8/8] Epoch complete (Time: {epoch_time:.1f}s)\n")

        if early_stopping(current_score):
            break

    # Mark fold completed and write training_state.json
    if fold_idx not in training_state.completed_folds:
        training_state.completed_folds.append(fold_idx)
    training_state.current_fold = fold_idx + 1 if fold_idx + 1 < config.n_splits else fold_idx
    training_state.last_epoch = config.num_epochs
    training_state.save(config.output_dir)
    print("✓ Fold completed")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2024 Full Competition Training Pipeline")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--all-folds", action="store_true", help="Train all 5 GroupKFold folds sequentially")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone model")

    args = parser.parse_args()
    config = Config()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.backbone is not None:
        config.backbone_name = args.backbone
        config.model_name = args.backbone

    if args.all_folds:
        train_full_ensemble(config, resume=args.resume)
    else:
        train(config, fold_idx=args.fold, resume=args.resume)
