"""
ISIC 2024 — Development Mode Training Script

Trains the FULL production FusionModel architecture (image backbone + metadata MLP)
on a small balanced subset for CPU-based demonstration.

Usage:
    python train_dev.py
    python train_dev.py --config configs/dev_config.yaml
    python train_dev.py --epochs 3 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import time
import datetime
from pathlib import Path
import sys
import os

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

# --- Project imports ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ISICDataset
from src.data.metadata import MetadataProcessor
from src.data.patient_features import enrich_metadata
from src.data.transforms import build_transforms
from src.models.fusion_model import FusionModel
from src.training.losses import get_loss_fn
from src.training.ema import ModelEMA
from src.evaluation.metrics import compute_pauc
from src.utils import ensure_dir, get_device, save_checkpoint, load_checkpoint, seed_everything, seed_worker


# =============================================================================
# Configuration Loader
# =============================================================================

def load_dev_config(config_path: str | Path = "configs/dev_config.yaml") -> dict:
    """Load development configuration from YAML file."""
    import yaml  # Lazy import; yaml is common in ML environments

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_config(args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI overrides."""
    cfg = {
        "mode": "development",
        "backbone_name": "tf_efficientnetv2_s",
        "image_size": 224,
        "use_metadata": True,
        "use_patient_features": True,
        "use_ugly_duckling": True,
        "metadata_mlp_hidden": 256,
        "metadata_mlp_output": 128,
        "num_epochs": 5,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": 1e-6,
        "loss_type": "focal",
        "focal_alpha": 0.50,
        "focal_gamma": 2.0,
        "early_stopping_patience": 3,
        "max_grad_norm": 1.0,
        "use_advanced_augs": False,
        "use_mixup": False,
        "use_cutmix": False,
        "use_ema": False,
        "ema_decay": 0.999,
        "use_fp16": False,
        "num_workers": 0,
        "pin_memory": False,
        "seed": 42,
        "train_csv": "data/dev_train.csv",
        "val_csv": "data/dev_validation.csv",
        "image_dir": "data/train-image",
        "checkpoint_dir": "outputs/checkpoints/dev",
        "evaluation_dir": "outputs/evaluation/dev",
        "target_column": "target",
        "image_id_column": "isic_id",
        "patient_column": "patient_id",
    }

    # Load from YAML
    yaml_cfg = load_dev_config(args.config)
    cfg.update(yaml_cfg)

    # CLI overrides
    if getattr(args, "epochs", None) is not None:
        cfg["num_epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        cfg["batch_size"] = args.batch_size
    if getattr(args, "lr", None) is not None:
        cfg["learning_rate"] = args.lr
    if getattr(args, "backbone", None) is not None:
        cfg["backbone_name"] = args.backbone
    if getattr(args, "image_size", None) is not None:
        cfg["image_size"] = args.image_size
    if getattr(args, "focal_alpha", None) is not None:
        cfg["focal_alpha"] = args.focal_alpha
    if getattr(args, "checkpoint_dir", None) is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if getattr(args, "evaluation_dir", None) is not None:
        cfg["evaluation_dir"] = args.evaluation_dir

    # Auto-adjust batch size based on available RAM
    if cfg["batch_size"] == "auto" or cfg["batch_size"] is None:
        cfg["batch_size"] = _auto_batch_size()

    return cfg


def _auto_batch_size() -> int:
    """Determine batch size based on available system RAM."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 32:
            return 16
        elif ram_gb >= 16:
            return 8
        else:
            return 4
    except ImportError:
        return 8


# =============================================================================
# Data Loading (uses exact same pipeline as production)
# =============================================================================

def build_dev_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, int, MetadataProcessor]:
    """Build train/val dataloaders from dev CSVs using the full metadata pipeline.

    This exercises the SAME code paths as production:
    - MetadataProcessor.fit_transform / transform
    - Patient feature enrichment
    - ISICDataset with metadata tensors
    - WeightedRandomSampler for class balance
    """
    train_csv = Path(cfg["train_csv"])
    val_csv = Path(cfg["val_csv"])
    image_dir = Path(cfg["image_dir"])

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Dev training CSV not found: {train_csv}\n"
            f"Run: python scripts/create_dev_splits.py"
        )
    if not val_csv.exists():
        raise FileNotFoundError(
            f"Dev validation CSV not found: {val_csv}\n"
            f"Run: python scripts/create_dev_splits.py"
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"  Train: {len(train_df)} samples "
          f"(benign={int((train_df[cfg['target_column']] == 0).sum())}, "
          f"malignant={int((train_df[cfg['target_column']] == 1).sum())})")
    print(f"  Val:   {len(val_df)} samples "
          f"(benign={int((val_df[cfg['target_column']] == 0).sum())}, "
          f"malignant={int((val_df[cfg['target_column']] == 1).sum())})")

    # --- Patient feature enrichment (same as production) ---
    if cfg["use_patient_features"]:
        print("  Computing patient features & ugly duckling scores...")
        train_df = enrich_metadata(train_df)
        val_df = enrich_metadata(val_df)

    # --- Metadata processing (same as production) ---
    processor = MetadataProcessor()
    train_meta_features = processor.fit_transform(train_df)
    val_meta_features = processor.transform(val_df)
    metadata_dim = train_meta_features.shape[1]

    # Save processor for inference compatibility
    processor_path = Path(cfg["checkpoint_dir"]) / "dev_metadata_processor.joblib"
    ensure_dir(processor_path.parent)
    processor.save(str(processor_path))
    print(f"  Metadata features: {metadata_dim} dims")
    print(f"  Metadata processor saved: {processor_path}")

    # --- Build datasets (same as production) ---
    train_dataset = ISICDataset(
        train_df,
        image_dir,
        transform=build_transforms(
            train=True,
            image_size=cfg["image_size"],
            use_advanced=cfg["use_advanced_augs"],
        ),
        target_col=cfg["target_column"],
        image_id_col=cfg["image_id_column"],
        metadata_tensor=train_meta_features,
    )
    val_dataset = ISICDataset(
        val_df,
        image_dir,
        transform=build_transforms(train=False, image_size=cfg["image_size"]),
        target_col=cfg["target_column"],
        image_id_col=cfg["image_id_column"],
        metadata_tensor=val_meta_features,
    )

    # --- Weighted sampler for class imbalance (same as production) ---
    train_targets = train_df[cfg["target_column"]].values
    unique, counts = np.unique(train_targets, return_counts=True)
    class_weights = counts.sum() / (len(unique) * counts)
    sample_weights = torch.tensor(
        [class_weights[int(t)] for t in train_targets],
        dtype=torch.float32,
    )

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=cfg["pin_memory"],
    )

    return train_loader, val_loader, metadata_dim, processor


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    ema: ModelEMA | None = None,
    epoch: int = 1,
    total_epochs: int = 5,
) -> float:
    """Train for one epoch. Same logic as production _train_one_epoch."""
    model.train()
    running_loss = 0.0
    running_samples = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        file=sys.stdout,
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for batch in pbar:
        images = batch["image"].to(device)
        metadata = batch["metadata"].to(device) if "metadata" in batch else None
        labels = batch["target"].to(device).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass (same as production)
        if metadata is not None:
            logits = model(images, metadata)
        else:
            logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["max_grad_norm"])
        optimizer.step()

        if ema is not None:
            ema.update(model)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_samples += batch_size
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{(running_loss / running_samples):.4f}"})

    return running_loss / max(running_samples, 1)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate and compute all metrics. Same logic as production validate."""
    model.eval()
    running_loss = 0.0
    running_samples = 0
    all_probs = []
    all_targets = []

    val_pbar = tqdm(
        dataloader,
        desc="Validation",
        file=sys.stdout,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    with torch.no_grad():
        for batch in val_pbar:
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device) if "metadata" in batch else None
            labels = batch["target"].to(device).float().unsqueeze(1)

            if metadata is not None:
                logits = model(images, metadata)
            else:
                logits = model(images)

            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size
            all_probs.append(probs)
            all_targets.append(labels.squeeze(-1).cpu().numpy())
            val_pbar.set_postfix({"val_loss": f"{(running_loss / running_samples):.4f}"})

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_probs)

    avg_loss = running_loss / max(running_samples, 1)

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pauc = compute_pauc(y_true, y_score, max_fpr=0.1)
    else:
        roc_auc = float("nan")
        pauc = float("nan")

    binary_preds = (y_score >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, binary_preds))
    prec = float(precision_score(y_true, binary_preds, zero_division=0))
    rec = float(recall_score(y_true, binary_preds, zero_division=0))
    f1 = float(f1_score(y_true, binary_preds, zero_division=0))

    return {
        "loss": avg_loss,
        "roc_auc": roc_auc,
        "pauc": pauc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "y_true": y_true,
        "y_score": y_score,
    }


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping to avoid overfitting."""
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.counter = 0
        self.stopped = False

    def __call__(self, current_score: float) -> bool:
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"  Early stopping triggered after {self.patience} epochs with no improvement.")
            self.stopped = True
            return True
        return False


# =============================================================================
# Evaluation & Plotting (Phase 6)
# =============================================================================

def generate_evaluation_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: dict,
    save_dir: Path,
):
    """Generate all evaluation plots and save metrics JSON."""
    ensure_dir(save_dir)

    # --- 1. Confusion Matrix ---
    binary_preds = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, binary_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # --- 2. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2.5, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.axvline(x=0.1, color="#FF5722", linestyle=":", alpha=0.7, label="FPR=0.1 (pAUC boundary)")
    ax.fill_between(fpr[fpr <= 0.1], tpr[fpr <= 0.1], alpha=0.15, color="#FF5722")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png", dpi=150)
    plt.close()

    # --- 3. Precision-Recall Curve ---
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_vals, precision_vals, color="#4CAF50", linewidth=2.5, label=f"AP = {ap:.4f}")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "precision_recall_curve.png", dpi=150)
    plt.close()

    # --- 4. Probability Histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_score[y_true == 0], bins=50, alpha=0.6, color="#2196F3", label="Benign", density=True)
    ax.hist(y_score[y_true == 1], bins=50, alpha=0.6, color="#F44336", label="Malignant", density=True)
    ax.set_title("Predicted Probability Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "probability_histogram.png", dpi=150)
    plt.close()

    # --- 5. Calibration Curve ---
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(prob_pred, prob_true, "s-", color="#9C27B0", linewidth=2, label="Model")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
        ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "calibration_curve.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate calibration curve: {e}")

    # --- 6. Save metrics JSON ---
    metrics_out = {k: v for k, v in metrics.items() if k not in ("y_true", "y_score")}
    with open(save_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"  Evaluation plots saved to: {save_dir}")


# =============================================================================
# Debug Model Quality (Phase 7)
# =============================================================================

def debug_model_quality(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    n_samples: int = 20,
) -> bool:
    """Run inference on N validation images and check prediction diversity.

    Returns True if predictions show meaningful variation, False if collapsed.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device) if "metadata" in batch else None
            labels = batch["target"].cpu().numpy()
            image_ids = batch["image_id"]

            if metadata is not None:
                logits = model(images, metadata)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            raw_logits = logits.squeeze(-1).cpu().numpy()

            for i in range(len(image_ids)):
                results.append({
                    "image_id": image_ids[i],
                    "ground_truth": int(labels[i]),
                    "raw_logit": float(raw_logits[i]),
                    "probability": float(probs[i]),
                    "prediction": int(probs[i] >= 0.5),
                    "confidence": float(abs(probs[i] - 0.5) * 2),
                })
                if len(results) >= n_samples:
                    break
            if len(results) >= n_samples:
                break

    # --- Print debug table ---
    print(f"\n{'='*90}")
    print(f"MODEL QUALITY DEBUG — {len(results)} validation samples")
    print(f"{'='*90}")
    print(f"{'Image ID':<20} {'Truth':>6} {'Logit':>10} {'Prob':>10} {'Pred':>6} {'Conf':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['image_id']:<20} {r['ground_truth']:>6} {r['raw_logit']:>10.4f} "
              f"{r['probability']:>10.4f} {r['prediction']:>6} {r['confidence']:>10.4f}")
    print("-" * 90)

    probs_arr = np.array([r["probability"] for r in results])
    logits_arr = np.array([r["raw_logit"] for r in results])
    print(f"\nProbability stats: min={probs_arr.min():.4f}, max={probs_arr.max():.4f}, "
          f"mean={probs_arr.mean():.4f}, std={probs_arr.std():.4f}")
    print(f"Logit stats:       min={logits_arr.min():.4f}, max={logits_arr.max():.4f}, "
          f"mean={logits_arr.mean():.4f}, std={logits_arr.std():.4f}")

    # Check if predictions are collapsed
    prob_range = probs_arr.max() - probs_arr.min()
    prob_std = probs_arr.std()

    if prob_range < 0.05 or prob_std < 0.02:
        print(f"\n[WARNING] Predictions appear COLLAPSED (range={prob_range:.4f}, std={prob_std:.4f})")
        print("[WARNING] The model is not producing meaningful probability variation.")
        return False
    else:
        print(f"\n[OK] Predictions show meaningful variation (range={prob_range:.4f}, std={prob_std:.4f})")
        return True


# =============================================================================
# Main Training Pipeline
# =============================================================================

def train_dev(cfg: dict) -> dict:
    """Full development training pipeline."""
    seed_everything(cfg["seed"])
    device = get_device()

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    evaluation_dir = Path(cfg["evaluation_dir"])
    ensure_dir(checkpoint_dir)
    ensure_dir(evaluation_dir)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("=" * 80)
    print("ISIC 2024 — DEVELOPMENT MODE TRAINING")
    print("=" * 80)
    print(f"  Mode:           {cfg['mode']}")
    print(f"  Backbone:       {cfg['backbone_name']}")
    print(f"  Image Size:     {cfg['image_size']}x{cfg['image_size']}")
    print(f"  Batch Size:     {cfg['batch_size']}")
    print(f"  Epochs:         {cfg['num_epochs']}")
    print(f"  Learning Rate:  {cfg['learning_rate']}")
    print(f"  Loss:           {cfg['loss_type']}")
    print(f"  Metadata:       {cfg['use_metadata']}")
    print(f"  Patient Feats:  {cfg['use_patient_features']}")
    print(f"  EMA:            {cfg['use_ema']}")
    print(f"  Device:         {device}")
    print(f"  Timestamp:      {timestamp}")
    print("=" * 80)

    # --- Build data loaders (full metadata pipeline) ---
    print("\nBuilding data loaders...")
    train_loader, val_loader, metadata_dim, processor = build_dev_loaders(cfg)

    # --- Build model (FusionModel, same as production) ---
    print(f"\nBuilding FusionModel: backbone={cfg['backbone_name']}, metadata_dim={metadata_dim}")
    model = FusionModel(
        backbone_name=cfg["backbone_name"],
        metadata_dim=metadata_dim,
        pretrained=True,
        metadata_hidden=cfg["metadata_mlp_hidden"],
        metadata_output=cfg["metadata_mlp_output"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # --- EMA (optional, same as production) ---
    ema = ModelEMA(model, decay=cfg["ema_decay"], device=device) if cfg["use_ema"] else None

    # --- Loss, optimizer, scheduler (same as production) ---
    criterion = get_loss_fn(
        cfg["loss_type"],
        alpha=cfg.get("focal_alpha", 0.75),
        gamma=cfg.get("focal_gamma", 2.0),
    )
    print(f"  Loss function: {criterion.__class__.__name__}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["num_epochs"],
        eta_min=cfg.get("scheduler_eta_min", 1e-6),
    )

    early_stopping = EarlyStopping(
        patience=cfg["early_stopping_patience"],
        min_delta=1e-4,
    )

    # --- Training loop ---
    best_roc_auc = float("-inf")
    best_pauc = float("-inf")
    history = []
    training_start = time.time()

    print(f"\n{'='*80}")
    print(f"Starting training for {cfg['num_epochs']} epochs...")
    print(f"{'='*80}\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg,
            ema=ema, epoch=epoch, total_epochs=cfg["num_epochs"],
        )

        # Validate
        eval_model = ema.module if ema is not None else model
        val_metrics = validate_epoch(eval_model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - training_start
        remaining = epoch_time * (cfg["num_epochs"] - epoch)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pauc": val_metrics["pauc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        history.append(epoch_result)

        # Display
        print(
            f"Epoch {epoch:>2d}/{cfg['num_epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"ROC-AUC={val_metrics['roc_auc']:.4f} | "
            f"pAUC={val_metrics['pauc']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={epoch_time:.1f}s | "
            f"elapsed={elapsed:.0f}s | "
            f"ETA={remaining:.0f}s"
        )

        # --- Save per-epoch checkpoint ---
        epoch_ckpt = {
            "mode": "development",
            "epoch": epoch,
            "model_name": cfg["backbone_name"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metadata_dim": metadata_dim,
            "use_metadata": cfg["use_metadata"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pauc": val_metrics["pauc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "train_loss": train_loss,
            "training_timestamp": timestamp,
            "config": cfg,
            "training_history": history,
        }
        if ema is not None:
            epoch_ckpt["ema_state_dict"] = ema.module.state_dict()

        save_checkpoint(epoch_ckpt, checkpoint_dir / f"epoch_{epoch}.pt")

        # --- Save best checkpoint ---
        current_score = val_metrics["roc_auc"]
        if np.isnan(current_score):
            current_score = float("-inf")

        if current_score > best_roc_auc:
            best_roc_auc = current_score
            best_pauc = val_metrics["pauc"] if not np.isnan(val_metrics["pauc"]) else 0.0
            save_checkpoint(epoch_ckpt, checkpoint_dir / "best_model.pt")
            print(f"  [BEST] New best ROC-AUC={best_roc_auc:.4f}, pAUC={best_pauc:.4f} -> saved best_model.pt")

        # Early stopping
        if early_stopping(current_score):
            break

    total_time = time.time() - training_start
    print(f"\n{'='*80}")
    print(f"Training complete! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best ROC-AUC: {best_roc_auc:.4f}")
    print(f"Best pAUC:    {best_pauc:.4f}")
    print(f"{'='*80}")

    # --- Save training history ---
    history_df = pd.DataFrame(history)
    history_df.to_csv(checkpoint_dir / "training_history.csv", index=False)

    # =========================================================================
    # Phase 6: Evaluation
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 6: GENERATING EVALUATION REPORTS")
    print(f"{'='*80}")

    # Load best checkpoint for final evaluation
    best_ckpt = load_checkpoint(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    final_metrics = validate_epoch(model, val_loader, criterion, device)
    generate_evaluation_plots(
        final_metrics["y_true"],
        final_metrics["y_score"],
        final_metrics,
        evaluation_dir,
    )

    print(f"\n  Final Evaluation Metrics:")
    print(f"    ROC-AUC:   {final_metrics['roc_auc']:.4f}")
    print(f"    pAUC@0.1:  {final_metrics['pauc']:.4f}")
    print(f"    Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"    Precision: {final_metrics['precision']:.4f}")
    print(f"    Recall:    {final_metrics['recall']:.4f}")
    print(f"    F1 Score:  {final_metrics['f1']:.4f}")

    # =========================================================================
    # Phase 7: Debug Model Quality
    # =========================================================================
    print(f"\n{'='*80}")
    print("PHASE 7: MODEL QUALITY DEBUGGING")
    print(f"{'='*80}")

    predictions_ok = debug_model_quality(model, val_loader, device, n_samples=20)

    if not predictions_ok:
        print("\n[ALERT] Model predictions are collapsed. Investigating...")
        print("  Possible causes:")
        print("  1. Learning rate too high or too low")
        print("  2. Batch size too small for BatchNorm layers")
        print("  3. Severe class imbalance in focal loss")
        print("  4. Insufficient training epochs")
        print("\n  The checkpoint has been saved. You can retry with different hyperparameters.")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("DEVELOPMENT TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"  Checkpoint dir:  {checkpoint_dir}")
    print(f"  Best model:      {checkpoint_dir / 'best_model.pt'}")
    print(f"  Evaluation dir:  {evaluation_dir}")
    print(f"  Training time:   {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best ROC-AUC:    {best_roc_auc:.4f}")
    print(f"  Best pAUC:       {best_pauc:.4f}")
    print(f"  Predictions OK:  {predictions_ok}")
    print(f"{'='*80}")

    return {
        "best_roc_auc": best_roc_auc,
        "best_pauc": best_pauc,
        "history": history,
        "checkpoint_dir": str(checkpoint_dir),
        "evaluation_dir": str(evaluation_dir),
        "training_time": total_time,
        "predictions_ok": predictions_ok,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ISIC 2024 — Development Mode Training (FusionModel on balanced subset)"
    )
    parser.add_argument("--config", type=str, default="configs/dev_config.yaml",
                        help="Path to dev config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone name")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Override focal loss alpha")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--evaluation-dir", type=str, default=None, help="Override evaluation output directory")
    args = parser.parse_args()

    cfg = get_config(args)
    train_dev(cfg)


if __name__ == "__main__":
    main()
