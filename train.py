from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.train import train, train_full_ensemble, resolve_resume_fold


def main():
    parser = argparse.ArgumentParser(
        description="ISIC 2024 Challenge — Official Competition Training CLI"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index for GroupKFold patient-level cross-validation (0 to 4)",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Sequentially train all 5 GroupKFold patient-level folds",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epoch count (default: config.num_epochs)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override model backbone architecture (e.g. tf_efficientnetv2_m, convnext_base)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )

    args = parser.parse_args()

    config = Config()

    # Apply command line parameter overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.backbone is not None:
        config.backbone_name = args.backbone
        config.model_name = args.backbone
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr

    print("=" * 80)
    print("ISIC 2024 COMPETITION TRAINING PIPELINE")
    print(f"  Project:         {config.project_name}")
    print(f"  Backbone:        {config.backbone_name}")
    print(f"  Image Size:      {config.image_size}x{config.image_size}")
    print(f"  Batch Size:      {config.batch_size}")
    print(f"  Epochs:          {config.num_epochs}")
    print(f"  Learning Rate:   {config.learning_rate}")
    print(f"  AMP FP16:        {config.use_fp16}")
    print(f"  EMA Enabled:     {getattr(config, 'use_ema', True)}")
    print(f"  Resume Training: {args.resume}")
    print("=" * 80 + "\n")

    if args.all_folds:
        print(f"Executing Full 5-Fold GroupKFold Training Pipeline...")
        train_full_ensemble(config, resume=args.resume)
    else:
        target_fold = args.fold
        if args.resume:
            target_fold, _ = resolve_resume_fold(config, requested_fold=args.fold, resume=True)
            if target_fold >= config.n_splits:
                print(f"[RESUME] All {config.n_splits} folds are already completed. Exiting cleanly.", flush=True)
                sys.exit(0)
        print(f"Executing Single-Fold Training Pipeline for Fold {target_fold}...")
        train(config, fold_idx=target_fold, resume=args.resume)


if __name__ == "__main__":
    main()
