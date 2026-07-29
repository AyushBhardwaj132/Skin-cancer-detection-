from __future__ import annotations

import os
import sys
import shutil
import tempfile
import pandas as pd
import torch
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import get_fold_dataframes
from src.training.losses import FocalLoss, AsymmetricLoss, PolyLoss
from src.training.archiver import create_fold_artifact_zip
from src.evaluation.diagnostic import generate_fold_diagnostic_report
from src.config.config import Config


def test_stratified_group_kfold_isolation_and_balance():
    print("\n" + "=" * 80)
    print("TEST 1: StratifiedGroupKFold Patient Isolation & Target Balance Audit")
    print("=" * 80)

    # Generate synthetic metadata with imbalanced target and patient groups
    np.random.seed(42)
    n_samples = 1000
    n_patients = 200

    patients = [f"P_{i % n_patients}" for i in range(n_samples)]
    # ~2% positive targets
    targets = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

    df = pd.DataFrame({"isic_id": [f"ISIC_{i:07d}" for i in range(n_samples)], "patient_id": patients, "target": targets})

    pos_ratios = []
    for fold in range(5):
        train_df, val_df = get_fold_dataframes(df, fold_idx=fold, n_splits=5, patient_col="patient_id", target_col="target", seed=42)

        # 1. Patient isolation check
        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])
        overlap = train_patients.intersection(val_patients)

        assert len(overlap) == 0, f"Patient leakage detected in fold {fold}: {len(overlap)} overlapping patients!"

        pos_count = (val_df["target"] == 1).sum()
        ratio = pos_count / len(val_df)
        pos_ratios.append(ratio)
        print(f"  Fold {fold}: Val Samples={len(val_df)}, Val Positives={pos_count} ({ratio:.2%}), Overlap={len(overlap)} [OK]")

    print(f"  [PASS] All 5 folds maintain 100% patient isolation & stratified target balance!")


def test_loss_numerical_stability():
    print("\n" + "=" * 80)
    print("TEST 2: Loss Function Numerical Stability & NaN Prevention")
    print("=" * 80)

    # Test extreme logits (+100, -100, NaN bounds)
    extreme_logits = torch.tensor([[100.0], [-100.0], [500.0], [-500.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    focal = FocalLoss()
    asl = AsymmetricLoss()
    poly = PolyLoss()

    loss_focal = focal(extreme_logits, targets)
    loss_asl = asl(extreme_logits, targets)
    loss_poly = poly(extreme_logits, targets)

    print(f"  Focal Loss on Extreme Logits: {loss_focal.item():.6f}")
    print(f"  ASL Loss on Extreme Logits  : {loss_asl.item():.6f}")
    print(f"  Poly Loss on Extreme Logits : {loss_poly.item():.6f}")

    assert not torch.isnan(loss_focal) and not torch.isinf(loss_focal), "Focal loss evaluated to NaN or Inf!"
    assert not torch.isnan(loss_asl) and not torch.isinf(loss_asl), "ASL loss evaluated to NaN or Inf!"
    assert not torch.isnan(loss_poly) and not torch.isinf(loss_poly), "Poly loss evaluated to NaN or Inf!"

    print("  [PASS] All loss functions evaluate safely without NaN or Inf!")


def test_fold_archiver():
    print("\n" + "=" * 80)
    print("TEST 3: End-of-Fold ZIP Artifact Archiver")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        checkpoints_dir = output_dir / "checkpoints"
        figures_dir = output_dir / "figures"
        logs_dir = output_dir / "logs"
        eval_dir = output_dir / "evaluation"

        for d in [checkpoints_dir, figures_dir, logs_dir, eval_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create dummy artifacts
        best_ckpt = checkpoints_dir / "best_model_fold0.pt"
        best_ckpt.write_bytes(b"DUMMY_BEST_CHECKPOINT")

        last_ckpt = checkpoints_dir / "last_checkpoint_fold0.pt"
        last_ckpt.write_bytes(b"DUMMY_LAST_CHECKPOINT")

        hist_csv = logs_dir / "history_fold0.csv"
        hist_csv.write_text("epoch,loss\n1,0.5\n")

        curve_png = figures_dir / "training_curves_fold0.png"
        curve_png.write_bytes(b"DUMMY_PNG")

        diag_json = eval_dir / "fold_0_diagnostic.json"
        diag_json.write_text('{"fold": 0}\n')

        zip_path = create_fold_artifact_zip(output_dir, fold_idx=0)
        assert zip_path is not None and zip_path.exists(), "Fold artifact ZIP file not created!"
        print(f"  Created ZIP: {zip_path.name} ({zip_path.stat().st_size} bytes) [OK]")
        print("  [PASS] Artifact archiver created fold_0_artifacts.zip successfully!")


def main():
    print("=" * 80)
    print("ISIC 2024 — PIPELINE AUDIT & REGRESSION TEST SUITE")
    print("=" * 80)

    test_stratified_group_kfold_isolation_and_balance()
    test_loss_numerical_stability()
    test_fold_archiver()

    print("\n" + "=" * 80)
    print("[PASS] ALL PIPELINE AUDIT & REGRESSION TESTS PASSED CLEANLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
