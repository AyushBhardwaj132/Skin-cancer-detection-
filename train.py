from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import Config
from src.training.runner import run_full_competition_pipeline
from src.train import run_debug_checkpoint_test


def main():
    parser = argparse.ArgumentParser(
        description="ISIC 2024 Challenge — Official Competition Training CLI"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold index (0 to 4). Omit to automatically train all 5 folds.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic checkpoint resuming and start training from scratch",
    )
    parser.add_argument(
        "--debug-checkpoint-test",
        action="store_true",
        help="Run tiny Kaggle verification mode and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose step-by-step debug logging",
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
    parser.add_argument(
        "--hf-test",
        action="store_true",
        help="Run Hugging Face authentication & upload self-test and exit",
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
    if args.debug_checkpoint_test:
        config.debug_checkpoint_test = True
    if args.debug:
        config.debug = True

    if args.hf_test:
        from src.training.hf_backup import HuggingFaceBackup
        hf = HuggingFaceBackup(repo_id=config.hf_repo_id)
        success = hf.perform_self_test(test_upload=True)
        sys.exit(0 if success else 1)
    elif config.debug_checkpoint_test:
        run_debug_checkpoint_test(config)
    else:
        run_full_competition_pipeline(
            config=config,
            requested_fold=args.fold,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
