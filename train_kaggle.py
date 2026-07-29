"""
ISIC 2024 — Kaggle GPU Training Entry Point

Executes 5-fold GroupKFold competition training on Kaggle GPU / Local CUDA environment.
Reuses existing production modules: Config, FusionModel, ISICDataset, run_full_competition_pipeline.
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
from src.training.runner import run_full_competition_pipeline
from src.train import run_debug_checkpoint_test


def setup_kaggle_hardware(config: Config) -> None:
    """Auto-detect CUDA GPU availability and configure hardware acceleration flags."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"  [CUDA DETECTED] GPU Hardware Acceleration Active: {device_name}")
        print(f"  [CUDA] Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
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

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISIC 2024 Challenge — Kaggle GPU Competition Training Entry Point"
    )
    parser.add_argument("--config", type=str, default="configs/kaggle_config.yaml", help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold index (0-4). Omit to train all 5 folds automatically.")
    parser.add_argument("--all-folds", action="store_true", help="Explicitly request training all 5 folds (default behavior)")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from existing checkpoint (default: True)")
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resuming and start from scratch")
    parser.add_argument("--debug-checkpoint-test", action="store_true", help="Run tiny Kaggle verification mode and exit")
    parser.add_argument("--debug", action="store_true", help="Enable verbose step-by-step debug logging")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--backbone", type=str, default=None, help="Override model backbone")
    parser.add_argument("--hf-test", action="store_true", help="Run Hugging Face authentication & upload self-test and exit")

    args = parser.parse_args()

    config = get_config(args)
    if args.debug_checkpoint_test:
        config.debug_checkpoint_test = True
    if args.debug:
        config.debug = True

    print("=" * 80)
    print("ISIC 2024 — KAGGLE GPU COMPETITION TRAINING PIPELINE")
    print("=" * 80)
    print(f"  Project:                {config.project_name}")
    print(f"  Resolved Data Dir:      {config.data_dir} [{'EXISTS' if config.data_dir.exists() else 'MISSING'}]")
    print(f"  Resolved Metadata Path: {config.train_metadata_path} [{'EXISTS' if config.train_metadata_path.exists() else 'MISSING'}]")
    print(f"  Resolved Output Dir:    {config.output_dir}")
    print(f"  Backbone:               {config.backbone_name}")
    print(f"  Image Size:             {config.image_size}x{config.image_size}")
    print(f"  Batch Size:             {config.batch_size}")
    print(f"  Epochs:                 {config.num_epochs}")
    print(f"  Learning Rate:          {config.learning_rate}")
    
    setup_kaggle_hardware(config)
    print("=" * 80 + "\n")

    if args.hf_test:
        from src.training.hf_backup import HuggingFaceBackup
        hf = HuggingFaceBackup(repo_id=config.hf_repo_id)
        success = hf.perform_self_test(test_upload=True)
        sys.exit(0 if success else 1)
    elif config.debug_checkpoint_test:
        run_debug_checkpoint_test(config)
    else:
        requested_fold = args.fold if not args.all_folds else None
        resume_flag = not args.no_resume
        run_full_competition_pipeline(
            config=config,
            requested_fold=requested_fold,
            resume=resume_flag,
        )


if __name__ == "__main__":
    main()
