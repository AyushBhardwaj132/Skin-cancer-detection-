"""
ISIC 2024 — Kaggle GPU Training Entry Point

Executes 5-fold GroupKFold competition training on Kaggle GPU / Local CUDA environment.
Reuses existing production modules: Config, FusionModel, ISICDataset, train_full_ensemble, train.
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.config import Config
from src.train import train, train_full_ensemble


def setup_kaggle_hardware(config: Config) -> None:
    """Auto-detect CUDA GPU availability and configure hardware acceleration flags."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"  [CUDA DETECTED] GPU Hardware Acceleration Active: {device_name}")
        print(f"  [CUDA] Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        # Enable GPU optimizations
        config.use_fp16 = True
        torch.backends.cudnn.benchmark = True
        print("  [GPU OPTIMIZATIONS] AMP FP16 Enabled | CuDNN Benchmark Enabled")
    else:
        print("  [CPU MODE] CUDA GPU not detected. Running standard CPU pipeline.")
        config.use_fp16 = False
        config.num_workers = 0


def get_config(args: argparse.Namespace) -> Config:
    """Load configuration from YAML file and apply CLI overrides."""
    config_path = args.config if args.config else "configs/kaggle_config.yaml"
    config = Config.from_yaml(config_path)

    print("After YAML load:")
    print(f"  backbone_name = {config.backbone_name}")
    print(f"  image_size    = {config.image_size}")
    print(f"  focal_alpha   = {config.focal_alpha}\n")

    # Apply CLI parameter overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.backbone is not None:
        config.backbone_name = args.backbone
        config.model_name = args.backbone

    print("After CLI overrides:")
    print(f"  backbone_name = {config.backbone_name}")
    print(f"  image_size    = {config.image_size}")
    print(f"  focal_alpha   = {config.focal_alpha}\n")

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISIC 2024 Challenge — Kaggle GPU Competition Training Entry Point"
    )
    parser.add_argument("--config", type=str, default="configs/kaggle_config.yaml", help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--all-folds", action="store_true", help="Train all 5 GroupKFold folds sequentially")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--backbone", type=str, default=None, help="Override model backbone")

    args = parser.parse_args()

    config = get_config(args)

    print("Immediately before train():")
    print(f"  backbone_name = {config.backbone_name}")
    print(f"  image_size    = {config.image_size}")
    print(f"  focal_alpha   = {config.focal_alpha}\n")

    print("=" * 80)

    print("ISIC 2024 — KAGGLE GPU COMPETITION TRAINING PIPELINE")
    print("=" * 80)
    print(f"  Project:                {config.project_name}")
    print(f"  Resolved Data Dir:      {config.data_dir} [{'EXISTS' if config.data_dir.exists() else 'MISSING'}]")
    print(f"  Resolved Metadata Path: {config.train_metadata_path} [{'EXISTS' if config.train_metadata_path.exists() else 'MISSING'}]")
    print(f"  Resolved HDF5 Path:     {config.train_image_hdf5_path} [{'EXISTS' if config.train_image_hdf5_path.exists() else 'MISSING'}]")
    print(f"  Resolved Output Dir:    {config.output_dir}")
    print(f"  Backbone:               {config.backbone_name}")
    print(f"  Image Size:             {config.image_size}x{config.image_size}")
    print(f"  Batch Size:             {config.batch_size}")
    print(f"  Epochs:                 {config.num_epochs}")
    print(f"  Learning Rate:          {config.learning_rate}")
    print(f"  Loss Function:          {config.loss_type} (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
    
    setup_kaggle_hardware(config)
    print("=" * 80 + "\n")

    if args.all_folds:
        print("Executing Full 5-Fold GroupKFold Ensemble Pipeline on Kaggle...")
        train_full_ensemble(config, resume=args.resume)
    else:
        print(f"Executing Single-Fold Training Pipeline for Fold {args.fold}...")
        train(config, fold_idx=args.fold, resume=args.resume)


if __name__ == "__main__":
    main()
