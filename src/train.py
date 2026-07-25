from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

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
from src.utils import ensure_dir, get_device, save_checkpoint, seed_everything
from src.validate import validate as run_validation


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
            train_df = enrich_metadata(train_df, patient_col="patient_id")
            val_df = enrich_metadata(val_df, patient_col="patient_id")

        # Fit metadata processor on train, transform both
        processor = MetadataProcessor()
        train_meta_features = processor.fit_transform(train_df)
        val_meta_features = processor.transform(val_df)
        metadata_dim = processor.get_feature_dim()

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
        target_column=config.target_column,
        image_id_column=config.image_id_column,
        metadata_features=train_meta_features,
    )
    val_dataset = ISICDataset(
        val_df,
        config.train_image_dir,
        transform=build_transforms(train=False, image_size=config.image_size),
        target_column=config.target_column,
        image_id_column=config.image_id_column,
        metadata_features=val_meta_features,
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
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, metadata_dim


def _train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    pbar_desc="",
    use_metadata: bool = True,
    use_mixup: bool = False,
    mixup_alpha: float = 0.4,
    use_cutmix: bool = False,
    cutmix_alpha: float = 1.0,
):
    """Train for one epoch with progress bar, optional MixUp/CutMix."""
    model.train()
    running_loss = 0.0
    running_samples = 0

    pbar = tqdm(dataloader, desc=pbar_desc, leave=False)
    for batch in pbar:
        images, metadata, labels = batch
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        # --- MixUp / CutMix ---
        if use_mixup and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            if use_metadata:
                logits = model(images, metadata)
            else:
                logits = model(images)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        elif use_cutmix and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
            optimizer.zero_grad(set_to_none=True)
            if use_metadata:
                logits = model(images, metadata)
            else:
                logits = model(images)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            optimizer.zero_grad(set_to_none=True)
            if use_metadata:
                logits = model(images, metadata)
            else:
                logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_samples += batch_size
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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


def train(config: Config | None = None, fold_idx: int = 0):
    """Phase 3 training: Fusion model with metadata, advanced losses, and MixUp/CutMix."""
    config = config or Config()
    seed_everything(config.seed)

    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.log_dir)
    ensure_dir(config.prediction_dir)
    ensure_dir(config.figures_dir)

    device = get_device()
    print(f"Device: {device}")

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
        ).to(device)
    else:
        print(f"Building image-only model: {config.model_name}")
        model = build_model(model_name=config.model_name, pretrained=True, num_classes=1).to(device)

    # --- Loss, optimizer, scheduler ---
    criterion = build_loss(
        config.loss_type,
        **({"alpha": config.focal_alpha, "gamma": config.focal_gamma}
           if config.loss_type == "focal" else {}),
    )
    print(f"Loss: {config.loss_type} ({criterion.__class__.__name__})")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=1e-4)

    best_pauc = float("-inf")
    best_auc = float("-inf")
    history: list[dict] = []

    print(f"\nTraining fold {fold_idx} for {config.num_epochs} epochs...")
    print(f"  MixUp={config.use_mixup}, CutMix={config.use_cutmix}, "
          f"Advanced Augs={config.use_advanced_augs}")
    print("-" * 80)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            pbar_desc=f"Epoch {epoch}/{config.num_epochs} [Train]",
            use_metadata=config.use_metadata,
            use_mixup=config.use_mixup,
            mixup_alpha=config.mixup_alpha,
            use_cutmix=config.use_cutmix,
            cutmix_alpha=config.cutmix_alpha,
        )

        val_metrics = run_validation(
            model, val_loader, criterion=criterion,
            device=device, use_metadata=config.use_metadata,
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pauc": val_metrics.get("pauc", float("nan")),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:>3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"roc_auc={val_metrics['roc_auc']:.4f} | "
            f"pAUC={val_metrics.get('pauc', float('nan')):.4f} | "
            f"lr={epoch_result['learning_rate']:.2e} | "
            f"time={epoch_time:.1f}s"
        )

        current_score = val_metrics.get("pauc", val_metrics["roc_auc"])
        if np.isnan(current_score):
            current_score = float("-inf")

        if current_score > best_pauc:
            best_pauc = current_score
            best_auc = val_metrics["roc_auc"]
            bb_dir = config.get_backbone_checkpoint_dir(config.backbone_name if config.use_metadata else config.model_name)
            checkpoint_path = bb_dir / f"best_model_fold{fold_idx}.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "fold": fold_idx,
                    "model_name": config.backbone_name if config.use_metadata else config.model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_pauc": best_pauc,
                    "best_val_auc": best_auc,
                    "metadata_dim": metadata_dim,
                    "use_metadata": config.use_metadata,
                    "config": {k: str(v) if isinstance(v, type(config.data_dir)) else v
                               for k, v in config.__dict__.items()},
                },
                checkpoint_path,
            )
            # Also save to root checkpoint dir for backwards compatibility
            save_checkpoint(
                {
                    "epoch": epoch,
                    "fold": fold_idx,
                    "model_name": config.backbone_name if config.use_metadata else config.model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_pauc": best_pauc,
                    "best_val_auc": best_auc,
                    "metadata_dim": metadata_dim,
                    "use_metadata": config.use_metadata,
                },
                config.checkpoint_dir / f"best_model_fold{fold_idx}.pt",
            )
            print(f"  ★ New best pAUC={best_pauc:.4f} — saved to {checkpoint_path}")

        if early_stopping(current_score):
            break

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = config.log_dir / f"training_history_{config.backbone_name}_fold{fold_idx}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\nTraining history saved to {history_path}")

    print(f"\nFold {fold_idx} complete: best_pAUC={best_pauc:.4f}, best_AUC={best_auc:.4f}")

    return {
        "history": history,
        "best_checkpoint": str(bb_dir / f"best_model_fold{fold_idx}.pt"),
        "best_val_pauc": best_pauc,
        "best_val_auc": best_auc,
        "metadata_dim": metadata_dim,
    }


def train_full_ensemble(config: Config | None = None, backbones: list[str] | None = None):
    """Train all ensemble backbones across all 5 GroupKFold folds (3 backbones x 5 folds = 15 models)."""
    config = config or Config()
    target_backbones = backbones or list(config.ensemble_backbones)
    
    print(f"\n{'='*80}")
    print(f"Starting Phase 4 Ensemble Training: {len(target_backbones)} backbones x {config.n_splits} folds = {len(target_backbones)*config.n_splits} models")
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
                res = train(cfg, fold_idx=fold)
                summary_results.append({
                    "backbone": bb,
                    "fold": fold,
                    "best_val_pauc": res["best_val_pauc"],
                    "best_val_auc": res["best_val_auc"],
                    "status": "Success",
                })
            except Exception as exc:
                print(f"ERROR training {bb} fold {fold}: {exc}")
                summary_results.append({
                    "backbone": bb,
                    "fold": fold,
                    "best_val_pauc": float("nan"),
                    "best_val_auc": float("nan"),
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

    """Compare different backbone architectures on the same fold.

    Runs short training with each backbone and reports pAUC.
    """
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

    # Summary
    results_df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("Backbone Comparison Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv(config.output_dir / "backbone_comparison.csv", index=False)
    return results_df


if __name__ == "__main__":
    train()
