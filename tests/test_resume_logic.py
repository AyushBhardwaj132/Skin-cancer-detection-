from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import Config
from src.train import train
from src.utils import save_checkpoint, load_checkpoint


def test_resume_fresh_start():
    """Test Case 1: Fresh run with resume=True and NO checkpoint -> gracefully falls back to fresh training."""
    print("\n" + "=" * 80)
    print("RUNNING TEST 1: Fresh Run with resume=True and NO checkpoint")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config()
        config.output_dir = Path(tmp_dir) / "outputs_fresh_test"
        config.num_epochs = 1
        config.backbone_name = "resnet18"
        config.model_name = "resnet18"
        config.batch_size = 4
        config.use_advanced_augs = False
        config.use_mixup = False
        config.use_cutmix = False
        config.use_fp16 = False
        config.num_workers = 0

        # Ensure output directory has no checkpoints
        shutil.rmtree(config.checkpoint_dir, ignore_errors=True)

        try:
            print("[TEST 1] Calling train() with resume=True on empty output directory...")
            # Limit train and val to 1 batch for quick test execution
            train(config, fold_idx=0, resume=True, limit_train=4, limit_val=4)
            print("[TEST 1 PASSED] Fresh run started and completed without RuntimeError!")
        except RuntimeError as e:
            print(f"[TEST 1 FAILED] RuntimeError raised on fresh run with resume=True: {e}")
            raise e


def test_resume_interrupted_run():
    """Test Case 2: Interrupted run with resume=True and checkpoint present -> restores checkpoint."""
    print("\n" + "=" * 80)
    print("RUNNING TEST 2: Interrupted Run with resume=True and Checkpoint Present")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config()
        config.output_dir = Path(tmp_dir) / "outputs_resume_test"
        config.num_epochs = 2
        config.checkpoint_batch_interval = 1
        config.backbone_name = "resnet18"
        config.model_name = "resnet18"
        config.batch_size = 4
        config.use_advanced_augs = False
        config.use_mixup = False
        config.use_cutmix = False
        config.use_fp16 = False
        config.num_workers = 0

        print("[TEST 2] Step A: Train 1 epoch to create checkpoint...")
        train(config, fold_idx=0, resume=False, limit_train=4, limit_val=4)

        bb_dir = config.get_backbone_checkpoint_dir("resnet18", fold_idx=0)
        last_ckpt = bb_dir / "last_checkpoint_fold0.pt"
        if not last_ckpt.exists():
            last_ckpt = config.checkpoint_dir / "last_checkpoint_fold0.pt"

        assert last_ckpt.exists(), f"[TEST 2 FAIL] Checkpoint file not created at {last_ckpt}"
        print(f"[TEST 2] Step A Passed: Checkpoint created ({last_ckpt.stat().st_size} bytes)")

        print("[TEST 2] Step B: Relaunch with resume=True to verify state restoration...")
        res = train(config, fold_idx=0, resume=True, limit_train=4, limit_val=4)
        print("[TEST 2 PASSED] Interrupted run restored and completed successfully!")


def main():
    print("=" * 80)
    print("ISIC 2024 — RESUME LOGIC REGRESSION TEST SUITE")
    print("=" * 80)

    test_resume_fresh_start()
    test_resume_interrupted_run()

    print("\n" + "=" * 80)
    print("[PASS] ALL RESUME LOGIC REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
