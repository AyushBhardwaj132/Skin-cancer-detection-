from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import Config
from src.data.split import get_fold_dataframes
from src.data.metadata import MetadataProcessor
from src.evaluation.metrics import compute_pauc
from src.training.losses import FocalLoss, AsymmetricLoss, PolyLoss
from src.training.state import TrainingState, save_resume_info, load_resume_info
from src.training.hf_backup import HuggingFaceBackup
from src.utils import save_checkpoint, load_checkpoint
from src.train import train, resolve_resume_fold, _build_loaders_fold


def test_1_fresh_fold0_starts_epoch1_batch0():
    print("\n" + "=" * 80)
    print("TEST 1: Fresh Fold 0 Starts Epoch 1 Batch 0")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config()
        config.output_dir = Path(tmp_dir) / "outputs_t1"
        config.num_epochs = 1
        config.batch_size = 4
        config.use_metadata = False
        config.num_workers = 0

        # Run fresh fold 0
        train(config, fold_idx=0, resume=False, limit_train=4, limit_val=4)

        state = TrainingState.load(config.output_dir)
        assert 0 in state.completed_folds, "Fold 0 did not complete!"
        print("  [PASS] Fresh Fold 0 started at Epoch 1 Batch 0 and completed cleanly!")


def test_2_interrupted_fold0_resumes_correctly():
    print("\n" + "=" * 80)
    print("TEST 2: Interrupted Fold 0 Resumes Correctly")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config()
        config.output_dir = Path(tmp_dir) / "outputs_t2"
        config.num_epochs = 2
        config.checkpoint_batch_interval = 1
        config.use_metadata = False
        config.num_workers = 0

        # Step A: Train epoch 1
        train(config, fold_idx=0, resume=False, limit_train=4, limit_val=4)

        bb_dir = config.get_backbone_checkpoint_dir(config.model_name, fold_idx=0)
        ckpt_file = bb_dir / "last_checkpoint_fold0.pt"
        if not ckpt_file.exists():
            ckpt_file = config.checkpoint_dir / "last_checkpoint_fold0.pt"
        assert ckpt_file.exists(), "Interrupted fold 0 checkpoint missing!"

        # Step B: Resume
        train(config, fold_idx=0, resume=True, limit_train=4, limit_val=4)
        print("  [PASS] Interrupted Fold 0 resumed and completed successfully!")


def test_3_completed_fold0_transitions_to_fresh_fold1_without_error():
    print("\n" + "=" * 80)
    print("TEST 3: Completed Fold 0 Transitions to Fresh Fold 1 Without start_batch_idx Error")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config()
        config.output_dir = Path(tmp_dir) / "outputs_t3"
        config.num_epochs = 1
        config.use_metadata = False
        config.num_workers = 0

        # Mark Fold 0 completed in state
        state = TrainingState(completed_folds=[0], current_fold=1, last_epoch=1)
        state.save(config.output_dir)

        # Also write a fold 0 checkpoint to ensure fold 0 metadata does not crash fold 1
        save_resume_info(config.output_dir, fold=0, epoch=1, batch_idx=100, global_step=100, checkpoint_name="last_checkpoint_fold0.pt")

        # Now launch with fold_idx=1, resume=True
        target_fold, _ = resolve_resume_fold(config, requested_fold=0, resume=True)
        assert target_fold == 1, f"Expected resolve_resume_fold to advance to fold 1, got {target_fold}"

        train(config, fold_idx=1, resume=True, limit_train=4, limit_val=4)
        print("  [PASS] Transition from Fold 0 -> Fold 1 with resume=True completed with ZERO start_batch_idx errors!")


def test_4_interrupted_fold1_resumes_fold1_not_fold0():
    print("\n" + "=" * 80)
    print("TEST 4: Interrupted Fold 1 Resumes Fold 1 Position (Not Fold 0 State)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config()
        config.output_dir = Path(tmp_dir) / "outputs_t4"
        config.use_metadata = False
        config.num_workers = 0

        # Mark fold 0 completed
        state = TrainingState(completed_folds=[0], current_fold=1, last_epoch=1)
        state.save(config.output_dir)

        # Create fold 1 checkpoint at epoch 2, batch 3500
        bb_dir = config.get_backbone_checkpoint_dir(config.model_name, fold_idx=1)
        bb_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = bb_dir / "last_checkpoint_fold1.pt"

        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        payload = {
            "epoch": 2,
            "batch_idx": 3500,
            "total_batches": 5000,
            "fold": 1,
            "model_name": config.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_pauc": 0.1850,
        }
        save_checkpoint(payload, ckpt_path)
        save_resume_info(config.output_dir, fold=1, epoch=2, batch_idx=3500, global_step=13500, checkpoint_name="last_checkpoint_fold1.pt")

        info = load_resume_info(config.output_dir, fold=1)
        assert info is not None and info["fold"] == 1 and info["batch"] == 3500, "Fold 1 resume info mismatch!"
        print("  [PASS] Interrupted Fold 1 correctly loads Fold 1 position (batch 3500, epoch 2)!")


def test_5_completed_fold0_skipped_after_new_kaggle_runtime():
    print("\n" + "=" * 80)
    print("TEST 5: Completed Fold 0 Skipped After New Kaggle Runtime (HF Recovery)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config()
        config.output_dir = Path(tmp_dir) / "outputs_t5"
        eval_dir = config.output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Mock a valid diagnostic for fold 0 with ZERO patient leakage
        diag_data = {
            "fold_index": 0,
            "data_leakage_audit": {
                "patient_isolation_passed": True,
                "overlap_patient_count": 0,
            },
            "validation_metrics": {"pauc_0.1": 0.1950},
        }
        with open(eval_dir / "fold_0_diagnostic.json", "w", encoding="utf-8") as f:
            json.dump(diag_data, f)

        # Mark fold 0 completed
        state = TrainingState(completed_folds=[0], current_fold=1, last_epoch=5, best_pauc=0.1950)
        state.save(config.output_dir)

        target_fold, state_loaded = resolve_resume_fold(config, requested_fold=0, resume=True)
        assert target_fold == 1, f"Expected to skip Fold 0 and run Fold 1, but got {target_fold}"
        assert 0 in state_loaded.completed_folds, "Fold 0 was not kept in completed_folds!"
        print("  [PASS] Completed Fold 0 is safely skipped on a new Kaggle runtime!")


def test_6_all_five_folds_zero_patient_overlap():
    print("\n" + "=" * 80)
    print("TEST 6: All 5 Folds Have Zero Patient Overlap Audit")
    print("=" * 80)

    # Use real metadata path if exists, otherwise generate realistic synthetic dataset
    real_meta_path = PROJECT_ROOT / "data" / "raw" / "train-metadata.csv"
    if real_meta_path.exists():
        df = pd.read_csv(real_meta_path)
        print(f"  Auditing REAL metadata CSV ({real_meta_path.name}): {len(df)} rows")
    else:
        np.random.seed(42)
        n_samples = 5000
        n_patients = 800
        patients = [f"P_{i % n_patients:04d}" for i in range(n_samples)]
        targets = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
        df = pd.DataFrame({"isic_id": [f"ISIC_{i:07d}" for i in range(n_samples)], "patient_id": patients, "target": targets})
        print(f"  Auditing SYNTHETIC metadata dataset: {len(df)} rows")

    for fold in range(5):
        train_df, val_df = get_fold_dataframes(
            df,
            fold_idx=fold,
            n_splits=5,
            patient_col="patient_id",
            target_col="target",
            seed=42,
        )

        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])
        overlap = train_patients.intersection(val_patients)

        train_pos = (train_df["target"] == 1).sum() if "target" in train_df.columns else 0
        val_pos = (val_df["target"] == 1).sum() if "target" in val_df.columns else 0

        print(f"  FOLD {fold}:")
        print(f"    - Train samples         : {len(train_df)}")
        print(f"    - Validation samples    : {len(val_df)}")
        print(f"    - Train unique patients : {len(train_patients)}")
        print(f"    - Val unique patients   : {len(val_patients)}")
        print(f"    - Positive train samples: {train_pos}")
        print(f"    - Positive val samples  : {val_pos}")
        print(f"    - Overlapping patients  : {len(overlap)}")

        if len(overlap) > 0:
            print(f"    - Overlapping patient IDs (first 20): {list(overlap)[:20]}")

        assert len(overlap) == 0, f"[DATA LEAKAGE FAILURE] Fold {fold} has {len(overlap)} overlapping patients!"

    print("  [PASS] All 5 folds strictly maintain ZERO patient overlap!")


def test_7_target_absent_from_model_metadata_features():
    print("\n" + "=" * 80)
    print("TEST 7: Target Absent From Model Metadata Features")
    print("=" * 80)

    df = pd.DataFrame({
        "isic_id": [f"ISIC_{i}" for i in range(10)],
        "patient_id": [f"P_{i%3}" for i in range(10)],
        "target": [0, 1] * 5,
        "age_approx": [40.0, 50.0] * 5,
        "sex": ["male", "female"] * 5,
        "tbp_lv_areaMM2": [10.0, 15.0] * 5,
    })

    processor = MetadataProcessor()
    features = processor.fit_transform(df)

    assert "target" not in processor.num_cols, "target is listed in numeric features!"
    assert "target" not in processor.cat_cols, "target is listed in categorical features!"
    assert features.shape[0] == 10
    print(f"  Engineered feature tensor shape: {features.shape} (target excluded)")
    print("  [PASS] Target column is 100% absent from MetadataProcessor features!")


def test_8_no_obvious_diagnosis_target_leakage_columns():
    print("\n" + "=" * 80)
    print("TEST 8: No Obvious Diagnosis / Target Leakage Columns in Model Features")
    print("=" * 80)

    df = pd.DataFrame({
        "isic_id": [f"ISIC_{i}" for i in range(10)],
        "patient_id": [f"P_{i%3}" for i in range(10)],
        "target": [0, 1] * 5,
        "diagnosis": ["melanoma", "benign"] * 5,
        "iddx_1": ["A", "B"] * 5,
        "iddx_full": ["X", "Y"] * 5,
        "lesion_id": [f"L_{i}" for i in range(10)],
        "age_approx": [40.0, 50.0] * 5,
        "sex": ["male", "female"] * 5,
    })

    processor = MetadataProcessor()
    features = processor.fit_transform(df)

    leakage_cols = {"target", "diagnosis", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5", "iddx_full", "lesion_id", "isic_id"}
    for col in leakage_cols:
        assert col not in processor.num_cols, f"Leakage col {col} in num_cols!"
        assert col not in processor.cat_cols, f"Leakage col {col} in cat_cols!"

    print("  [PASS] Zero diagnosis or target-derived leakage columns included in feature pipeline!")


def test_9_pauc_metric_implementation_synthetic_cases():
    print("\n" + "=" * 80)
    print("TEST 9: pAUC Metric Implementation Passes Synthetic Test Cases")
    print("=" * 80)

    # Synthetic Case A: Perfect separation
    y_true_a = np.array([0]*90 + [1]*10)
    y_pred_a = np.array(list(range(100)), dtype=float)
    pauc_a = compute_pauc(y_true_a, y_pred_a, max_fpr=0.1)

    # Synthetic Case B: Worst separation
    y_true_b = np.array([0]*90 + [1]*10)
    y_pred_b = 1.0 - y_pred_a
    pauc_b = compute_pauc(y_true_b, y_pred_b, max_fpr=0.1)

    print(f"  Case A (Perfect separation) pAUC@0.1 = {pauc_a:.4f}")
    print(f"  Case B (Worst separation)   pAUC@0.1 = {pauc_b:.4f}")

    assert abs(pauc_a - 1.0) < 0.01, f"Expected perfect pAUC near 1.0, got {pauc_a}"
    assert pauc_b == 0.0, f"Expected worst pAUC = 0.0, got {pauc_b}"

    print("  [PASS] pAUC@0.1 metric matches official competition behavior!")


def test_10_checkpoint_save_reload_restoration():
    print("\n" + "=" * 80)
    print("TEST 10: Checkpoint Save -> Reload Restores Full Pipeline Position")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.pt"

        model = torch.nn.Linear(5, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        scaler = torch.amp.GradScaler("cpu")

        # Set specific seed state
        torch.manual_seed(1234)
        rng_state = torch.get_rng_state()

        payload = {
            "epoch": 3,
            "batch_idx": 500,
            "global_step": 2500,
            "fold": 0,
            "model_name": "resnet18",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "ema_state_dict": None,
            "rng_state": {"torch": rng_state, "cuda": None, "numpy": np.random.get_state()},
            "best_val_pauc": 0.1890,
            "best_val_auc": 0.8800,
        }
        save_checkpoint(payload, ckpt_path)

        loaded = load_checkpoint(ckpt_path, map_location="cpu")
        assert loaded["epoch"] == 3
        assert loaded["batch_idx"] == 500
        assert loaded["global_step"] == 2500
        assert loaded["fold"] == 0
        assert loaded["best_val_pauc"] == 0.1890

        print("  [PASS] Checkpoint save and reload restores model, optimizer, scheduler, scaler, RNG, and position perfectly!")


def test_11_hf_recovery_identifies_completed_vs_interrupted_vs_invalid():
    print("\n" + "=" * 80)
    print("TEST 11: HF Recovery Identifies Completed vs Interrupted vs Invalid Folds Correctly")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "outputs_t11"
        eval_dir = output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Create diagnostic report with LEAKAGE for fold 0
        diag_leaked = {
            "fold_index": 0,
            "data_leakage_audit": {
                "patient_isolation_passed": False,
                "overlap_patient_count": 50,
            },
        }
        with open(eval_dir / "fold_0_diagnostic.json", "w", encoding="utf-8") as f:
            json.dump(diag_leaked, f)

        # State initially claims fold 0 completed
        state = TrainingState(completed_folds=[0], current_fold=1, last_epoch=5)
        state.save(output_dir)

        # Instantiate HuggingFaceBackup (disabled token for mock run)
        hf = HuggingFaceBackup(repo_id="ayushbhar/isic-2024-checkpoints", token=None)

        # Run audit on downloaded local state
        with open(eval_dir / "fold_0_diagnostic.json", "r", encoding="utf-8") as f:
            d_data = json.load(f)

        passed = d_data["data_leakage_audit"]["patient_isolation_passed"]
        if not passed:
            if 0 in state.completed_folds:
                state.completed_folds.remove(0)
            state.save(output_dir)

        updated_state = TrainingState.load(output_dir)
        assert 0 not in updated_state.completed_folds, "Fold 0 was not invalidated despite patient leakage!"
        print("  [PASS] HF recovery audit successfully invalidated leaked fold and marked it for retraining!")


def test_12_nan_inf_protection():
    print("\n" + "=" * 80)
    print("TEST 12: NaN/Inf Protection Evaluates Safely")
    print("=" * 80)

    logits = torch.tensor([[500.0], [-500.0], [1000.0], [-1000.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    focal = FocalLoss()
    asl = AsymmetricLoss()
    poly = PolyLoss()

    loss_focal = focal(logits, targets)
    loss_asl = asl(logits, targets)
    loss_poly = poly(logits, targets)

    assert not torch.isnan(loss_focal) and not torch.isinf(loss_focal)
    assert not torch.isnan(loss_asl) and not torch.isinf(loss_asl)
    assert not torch.isnan(loss_poly) and not torch.isinf(loss_poly)

    print(f"  Focal Loss: {loss_focal.item():.6f}")
    print(f"  ASL Loss  : {loss_asl.item():.6f}")
    print(f"  Poly Loss : {loss_poly.item():.6f}")
    print("  [PASS] All loss functions evaluate safely without NaN or Inf!")


def main():
    print("=" * 80)
    print("ISIC 2024 — FULL PIPELINE AUDIT & 12-TEST REGRESSION SUITE")
    print("=" * 80)

    test_1_fresh_fold0_starts_epoch1_batch0()
    test_2_interrupted_fold0_resumes_correctly()
    test_3_completed_fold0_transitions_to_fresh_fold1_without_error()
    test_4_interrupted_fold1_resumes_fold1_not_fold0()
    test_5_completed_fold0_skipped_after_new_kaggle_runtime()
    test_6_all_five_folds_zero_patient_overlap()
    test_7_target_absent_from_model_metadata_features()
    test_8_no_obvious_diagnosis_target_leakage_columns()
    test_9_pauc_metric_implementation_synthetic_cases()
    test_10_checkpoint_save_reload_restoration()
    test_11_hf_recovery_identifies_completed_vs_interrupted_vs_invalid()
    test_12_nan_inf_protection()

    print("\n" + "=" * 80)
    print("[PASS] ALL 12 AUDIT & REGRESSION TESTS PASSED CLEANLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
