import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.training.state import TrainingState
from src.utils import load_checkpoint, save_checkpoint


def test_training_state_schema_and_serialization(tmp_path):
    """Test TrainingState json schema, default values, save and load behavior."""
    state_dir = tmp_path / "outputs"
    
    # 1. Initial empty state
    state = TrainingState.load(state_dir)
    assert state.completed_folds == []
    assert state.current_fold == 0
    assert state.last_epoch == 0
    assert state.best_pauc == 0.0

    # 2. Update and save state
    state.completed_folds = [0, 1]
    state.current_fold = 2
    state.last_epoch = 7
    state.best_pauc = 0.1979
    saved_path = state.save(state_dir)

    assert saved_path.exists()
    assert saved_path.name == "training_state.json"

    # 3. Read back file and check exact JSON structure
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data == {
        "completed_folds": [0, 1],
        "current_fold": 2,
        "last_epoch": 7,
        "best_pauc": 0.1979,
    }

    # 4. Load via TrainingState.load
    loaded = TrainingState.load(state_dir)
    assert loaded.completed_folds == [0, 1]
    assert loaded.current_fold == 2
    assert loaded.last_epoch == 7
    assert loaded.best_pauc == 0.1979


def test_checkpoint_payload_contents(tmp_path):
    """Test that saved checkpoint payload contains optimizer, scheduler, scaler, EMA, metrics, config."""
    ckpt_path = tmp_path / "last_checkpoint_fold0.pt"
    
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    payload = {
        "epoch": 3,
        "fold": 0,
        "model_name": "tf_efficientnetv2_s",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None,
        "ema_state_dict": None,
        "best_val_pauc": 0.1850,
        "best_val_auc": 0.8500,
        "val_pauc": 0.1850,
        "val_auc": 0.8500,
        "val_loss": 0.450,
        "metrics": {"val_pauc": 0.1850, "val_auc": 0.8500},
        "config": {"image_size": 224, "batch_size": 8},
    }

    save_checkpoint(payload, ckpt_path)
    assert ckpt_path.exists()

    # Load with PyTorch 2.6 compatibility (load_checkpoint defaults to weights_only=False)
    loaded_ckpt = load_checkpoint(ckpt_path)

    assert loaded_ckpt["epoch"] == 3
    assert loaded_ckpt["fold"] == 0
    assert "optimizer_state_dict" in loaded_ckpt
    assert "scheduler_state_dict" in loaded_ckpt
    assert "config" in loaded_ckpt
    assert loaded_ckpt["best_val_pauc"] == 0.1850


def test_resume_skips_completed_folds(tmp_path):
    """Test that TrainingState correctly indicates completed folds to be skipped."""
    state_dir = tmp_path / "outputs"
    state = TrainingState(completed_folds=[0, 1], current_fold=2, last_epoch=10, best_pauc=0.25)
    state.save(state_dir)

    loaded_state = TrainingState.load(state_dir)
    assert 0 in loaded_state.completed_folds
    assert 1 in loaded_state.completed_folds
    assert 2 not in loaded_state.completed_folds


def test_per_epoch_last_checkpoint_and_resume_flow(tmp_path):
    """Simulate training interruption at epoch 2 and resuming from epoch 3."""
    out_dir = tmp_path / "outputs"
    ckpt_dir = out_dir / "checkpoints" / "efficientnetv2_s"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Simulate saving epoch 1 and 2 last checkpoints
    state = TrainingState(completed_folds=[], current_fold=0, last_epoch=2, best_pauc=0.1500)
    state.save(out_dir)

    model = torch.nn.Linear(5, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    last_ckpt = ckpt_dir / "last_checkpoint_fold0.pt"
    payload = {
        "epoch": 2,
        "fold": 0,
        "model_name": "tf_efficientnetv2_s",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_pauc": 0.1500,
        "best_val_auc": 0.8200,
        "val_pauc": 0.1500,
        "val_auc": 0.8200,
    }
    save_checkpoint(payload, last_ckpt)

    # 2. Simulate resuming training: load state & last checkpoint
    reloaded_state = TrainingState.load(out_dir)
    assert reloaded_state.last_epoch == 2

    ckpt = load_checkpoint(last_ckpt)
    assert ckpt["epoch"] == 2
    resume_start_epoch = ckpt["epoch"] + 1
    assert resume_start_epoch == 3

    # 3. Simulate completing fold 0 at epoch 5
    reloaded_state.completed_folds.append(0)
    reloaded_state.current_fold = 1
    reloaded_state.last_epoch = 5
    reloaded_state.save(out_dir)

    final_state = TrainingState.load(out_dir)
    assert final_state.completed_folds == [0]
    assert final_state.current_fold == 1
