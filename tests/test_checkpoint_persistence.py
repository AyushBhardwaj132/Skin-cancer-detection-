import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.training.state import TrainingState
from src.utils import save_checkpoint, load_checkpoint, sync_file


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.randn(3, 32, 32),
            "target": torch.tensor(idx % 2, dtype=torch.float32),
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32 * 32 * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1))


def test_physical_checkpoint_persistence(tmp_path: Path):
    """Verify save_checkpoint and TrainingState.save physically write to disk, sync, and reload."""
    output_dir = tmp_path / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test save_checkpoint with dummy model & optimizer state
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = checkpoint_dir / "last_checkpoint_fold0.pt"

    payload = {
        "epoch": 1,
        "batch_idx": 5,
        "fold": 0,
        "model_name": "dummy_model",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_pauc": 0.25,
    }

    saved_path = save_checkpoint(payload, ckpt_path)

    # Assert 1: Physical existence check using os.path.exists
    assert os.path.exists(saved_path), f"Checkpoint file does not exist: {saved_path}"
    assert saved_path.is_file()

    # Assert 2: Size > 0
    size_bytes = os.path.getsize(saved_path)
    assert size_bytes > 0, f"Checkpoint size is 0 bytes: {saved_path}"

    # Assert 3: torch.load succeeds and returns correct keys/tensors
    reloaded = torch.load(saved_path, map_location="cpu", weights_only=False)
    assert reloaded["epoch"] == 1
    assert reloaded["batch_idx"] == 5
    assert reloaded["best_val_pauc"] == 0.25
    assert "model_state_dict" in reloaded
    assert "optimizer_state_dict" in reloaded

    # Assert 4: TrainingState.save physically writes training_state.json
    state = TrainingState(completed_folds=[], current_fold=0, last_epoch=1, last_batch_idx=5, best_pauc=0.25)
    state_file = state.save(output_dir)

    assert os.path.exists(state_file), f"training_state.json does not exist: {state_file}"
    assert os.path.getsize(state_file) > 0

    with open(state_file, "r", encoding="utf-8") as f:
        state_data = json.load(f)
    assert state_data["current_fold"] == 0
    assert state_data["last_epoch"] == 1
    assert state_data["last_batch_idx"] == 5
    assert state_data["best_pauc"] == 0.25

    # Assert 5: Directory contains expected files
    dir_files = [f.name for f in checkpoint_dir.iterdir() if f.is_file()]
    assert "last_checkpoint_fold0.pt" in dir_files
    assert os.path.exists(output_dir / "training_state.json")


def test_fsync_file_verification(tmp_path: Path):
    """Verify sync_file flushes kernel buffer to disk for existing files."""
    test_file = tmp_path / "test_sync.bin"
    test_file.write_bytes(b"test_checkpoint_sync_bytes" * 100)

    synced_path = sync_file(test_file)
    assert os.path.exists(synced_path)
    assert os.path.getsize(synced_path) == 2600

    # Non-existent file sync must raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot sync file because it does not exist"):
        sync_file(tmp_path / "non_existent_file.pt")


def test_strict_resume_failure_when_missing(tmp_path: Path):
    """Verify train() raises RuntimeError when --resume is requested but no checkpoint exists."""
    from src.train import train

    config = Config.from_yaml("configs/kaggle_config.yaml")
    config.output_dir = tmp_path / "outputs"

    with pytest.raises(RuntimeError, match="--resume requested, but no valid checkpoint file was found"):
        train(config, fold_idx=0, resume=True)


def test_run_debug_checkpoint_test_mode(tmp_path: Path):
    """Verify run_debug_checkpoint_test executes 1 tiny epoch, saves, reloads, and resumes."""
    from src.train import run_debug_checkpoint_test

    config = Config.from_yaml("configs/kaggle_config.yaml")
    config.output_dir = tmp_path / "outputs"

    run_debug_checkpoint_test(config)

    state_file = config.output_dir / "training_state.json"
    last_ckpt = config.checkpoint_dir / "last_checkpoint_fold0.pt"

    assert os.path.exists(state_file)
    assert os.path.exists(last_ckpt)
    assert os.path.getsize(state_file) > 0
    assert os.path.getsize(last_ckpt) > 0

