"""
Smoke test: 1000 train / 500 val samples, 1 epoch, exercise checkpointing & state persistence.
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.train import train
from src.utils import ensure_dir, seed_everything, sync_file


TRAIN_LIMIT = 1000
VAL_LIMIT = 500


def main():
    # --- Config ---
    config = Config.from_yaml("configs/kaggle_config.yaml")
    config.num_epochs = 1
    config.num_workers = 0  # safe for quick CPU run
    config.use_advanced_augs = False
    config.use_mixup = False
    config.use_cutmix = False
    config.use_fp16 = False
    config.checkpoint_batch_interval = 10  # Force intra-epoch checkpoint every 10 batches
    seed_everything(config.seed)

    print("=" * 80)
    print("RUNNING SMOKE TEST VIA PRODUCTION TRAIN PIPELINE")
    print("=" * 80)
    print(f"Train limit: {TRAIN_LIMIT}, Val limit: {VAL_LIMIT}")
    print(f"Intra-epoch checkpoint interval: {config.checkpoint_batch_interval}")

    # Run production training pipeline on small subset
    results = train(config, fold_idx=0, limit_train=TRAIN_LIMIT, limit_val=VAL_LIMIT)

    # Save smoke_test_fold0.pt artifact for backward compatibility
    ensure_dir(config.checkpoint_dir)
    smoke_ckpt_path = config.checkpoint_dir / "smoke_test_fold0.pt"
    last_ckpt_path = config.checkpoint_dir / "last_checkpoint_fold0.pt"

    if last_ckpt_path.exists():
        shutil.copy2(last_ckpt_path, smoke_ckpt_path)
        sync_file(smoke_ckpt_path)
        print(f"\n[ARTIFACT] Copied last checkpoint to: {smoke_ckpt_path}")

    # Verify physical file existence
    state_file = config.output_dir / "training_state.json"
    print("\n" + "=" * 80)
    print("SMOKE TEST PHYSICAL FILE VERIFICATION")
    print("=" * 80)
    print(f"  training_state.json        : {'EXISTS' if state_file.exists() else 'MISSING'} ({state_file})")
    print(f"  last_checkpoint_fold0.pt   : {'EXISTS' if last_ckpt_path.exists() else 'MISSING'} ({last_ckpt_path})")
    print(f"  smoke_test_fold0.pt        : {'EXISTS' if smoke_ckpt_path.exists() else 'MISSING'} ({smoke_ckpt_path})")
    print("=" * 80 + "\n")

    if not state_file.exists() or not last_ckpt_path.exists():
        raise RuntimeError("Smoke test failed: training_state.json or last_checkpoint_fold0.pt missing!")

    print("[DONE] Smoke test completed successfully.")


if __name__ == "__main__":
    main()
