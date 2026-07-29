from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.config.config import Config
from src.data.dataset import ISICDataset
from src.data.split import get_fold_dataframes
from src.data.transforms import build_transforms
from src.data.caching import MetadataCacheManager
from src.data.metadata import MetadataProcessor
from src.models.fusion_model import FusionModel
from src.models.model import build_model
from src.metrics import compute_pauc
from src.utils import get_device, load_checkpoint, ensure_dir


def generate_oof_predictions(config: Config) -> tuple[pd.DataFrame, float, float]:
    """Generates Out-Of-Fold (OOF) predictions across all 5 folds using best fold models.
    
    Returns:
        (oof_df, cv_pauc, cv_auc)
    """
    device = get_device()
    print("\n" + "=" * 80)
    print("COMPUTING OUT-OF-FOLD (OOF) PREDICTIONS ACROSS ALL FOLDS")
    print("=" * 80)

    # Load enriched metadata from cache or compute
    full_meta_df = MetadataCacheManager.load_or_compute_enriched_metadata(
        config.train_metadata_path, config, verbose=False
    )

    oof_frames = []

    for fold_idx in range(config.n_splits):
        bb_dir = config.get_backbone_checkpoint_dir(config.backbone_name, fold_idx=fold_idx)
        best_ckpt = bb_dir / f"best_model_fold{fold_idx}.pt"
        if not best_ckpt.exists():
            # Fallback to root checkpoint directory
            best_ckpt = config.checkpoint_dir / f"best_model_fold{fold_idx}.pt"
        
        if not best_ckpt.exists():
            print(f"  [WARN] Best checkpoint for Fold {fold_idx} missing at {best_ckpt}. Skipping OOF for Fold {fold_idx}.")
            continue

        print(f"  Evaluating Fold {fold_idx} OOF using checkpoint: {best_ckpt.name}...")
        ckpt = load_checkpoint(best_ckpt, map_location=device)
        metadata_dim = ckpt.get("metadata_dim", 1)

        _, val_df = get_fold_dataframes(
            config.train_metadata_path,
            fold_idx=fold_idx,
            n_splits=config.n_splits,
        )

        # Merge preprocessed features if available
        if config.use_metadata:
            val_df = full_meta_df.iloc[val_df.index].reset_index(drop=True)
            processor = MetadataProcessor()
            val_meta_features = processor.fit_transform(val_df)
            if val_meta_features.shape[1] > metadata_dim:
                val_meta_features = val_meta_features[:, :metadata_dim]
            elif val_meta_features.shape[1] < metadata_dim:
                pad = metadata_dim - val_meta_features.shape[1]
                val_meta_features = np.pad(val_meta_features, ((0,0),(0,pad)), mode='constant')
        else:
            val_meta_features = None

        val_dataset = ISICDataset(
            val_df,
            config.train_image_dir,
            transform=build_transforms(train=False, image_size=config.image_size),
            target_col=config.target_column,
            image_id_col=config.image_id_column,
            metadata_tensor=val_meta_features,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model_name = ckpt.get("model_name", config.backbone_name)
        if config.use_metadata and metadata_dim > 0:
            model = FusionModel(
                backbone_name=model_name,
                metadata_dim=metadata_dim,
                pretrained=False,
            ).to(device)
        else:
            model = build_model(model_name=model_name, pretrained=False, num_classes=1).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                meta = batch["metadata"].to(device) if (config.use_metadata and "metadata" in batch) else None
                logits = model(images, meta) if (config.use_metadata and meta is not None) else model(images)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                fold_preds.append(probs)
                fold_targets.append(batch["target"].numpy())

        preds_arr = np.concatenate(fold_preds, axis=0)
        targets_arr = np.concatenate(fold_targets, axis=0)

        fold_pauc = compute_pauc(targets_arr, preds_arr, max_fpr=0.1)
        print(f"    Fold {fold_idx} OOF pAUC @ 0.1: {fold_pauc:.4f}")

        fold_result_df = pd.DataFrame({
            config.image_id_column: val_df[config.image_id_column],
            config.target_column: targets_arr,
            "oof_pred_prob": preds_arr,
            "fold": fold_idx,
        })
        oof_frames.append(fold_result_df)

    if not oof_frames:
        print("  [WARN] No fold checkpoints available to generate OOF predictions.")
        return pd.DataFrame(), 0.0, 0.0

    oof_df = pd.concat(oof_frames, axis=0).reset_index(drop=True)
    
    cv_pauc = compute_pauc(oof_df[config.target_column].values, oof_df["oof_pred_prob"].values, max_fpr=0.1)
    cv_auc = float(roc_auc_score(oof_df[config.target_column].values, oof_df["oof_pred_prob"].values))

    oof_out_path = config.prediction_dir / "oof_predictions.csv"
    ensure_dir(oof_out_path.parent)
    oof_df.to_csv(oof_out_path, index=False)

    print("\n" + "=" * 80)
    print(f"OVERALL 5-FOLD CROSS-VALIDATION RESULTS:")
    print(f"  Total OOF Predictions : {len(oof_df)}")
    print(f"  Overall 5-Fold pAUC   : {cv_pauc:.4f}")
    print(f"  Overall 5-Fold ROC-AUC: {cv_auc:.4f}")
    print(f"  OOF CSV Saved To      : {oof_out_path}")
    print("=" * 80 + "\n")

    return oof_df, cv_pauc, cv_auc
