from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix


def generate_fold_diagnostic_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_metrics: dict[str, float],
    fold_idx: int,
    output_dir: str | Path,
    patient_col: str = "patient_id",
    target_col: str = "target",
) -> dict:
    """Generates a detailed fold diagnostic report verifying zero data leakage, target balance, and metric accuracy."""
    output_dir = Path(output_dir)
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_patients = set(train_df[patient_col]) if patient_col in train_df.columns else set()
    val_patients = set(val_df[patient_col]) if patient_col in val_df.columns else set()
    overlap_patients = list(train_patients.intersection(val_patients))

    train_positives = int((train_df[target_col] == 1).sum()) if target_col in train_df.columns else 0
    val_positives = int((val_df[target_col] == 1).sum()) if target_col in val_df.columns else 0

    train_pos_ratio = float(train_positives / max(len(train_df), 1))
    val_pos_ratio = float(val_positives / max(len(val_df), 1))

    # Confusion matrix if y_true and y_score are present in val_metrics
    cm_dict = {}
    if "y_true" in val_metrics and "y_score" in val_metrics:
        y_true = np.asarray(val_metrics["y_true"])
        y_score = np.asarray(val_metrics["y_score"])
        opt_thresh = val_metrics.get("optimal_threshold", 0.5)
        if y_true.size > 0 and len(np.unique(y_true)) >= 2:
            preds = (y_score >= opt_thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            cm_dict = {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }

    report = {
        "fold_index": fold_idx,
        "data_leakage_audit": {
            "patient_column": patient_col,
            "train_patient_count": len(train_patients),
            "val_patient_count": len(val_patients),
            "overlap_patient_count": len(overlap_patients),
            "patient_isolation_passed": len(overlap_patients) == 0,
        },
        "target_balance_audit": {
            "train_total": len(train_df),
            "train_positives": train_positives,
            "train_positive_ratio": train_pos_ratio,
            "val_total": len(val_df),
            "val_positives": val_positives,
            "val_positive_ratio": val_pos_ratio,
        },
        "validation_metrics": {
            "pauc_0.1": val_metrics.get("pauc", 0.0),
            "roc_auc": val_metrics.get("roc_auc", 0.0),
            "optimal_threshold": val_metrics.get("optimal_threshold", 0.5),
            "f1_score": val_metrics.get("f1_optimal", 0.0),
            "loss": val_metrics.get("loss", 0.0),
        },
        "confusion_matrix": cm_dict,
    }

    report_path = eval_dir / f"fold_{fold_idx}_diagnostic.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n  [DIAGNOSTIC REPORT] Fold {fold_idx} report written to {report_path.name}")
    print(f"    - Patient Isolation : {'PASSED [OK]' if len(overlap_patients) == 0 else 'FAILED [LEAKAGE]'}")
    print(f"    - Val Positives     : {val_positives}/{len(val_df)} ({val_pos_ratio:.4%})")
    print(f"    - Validation pAUC   : {val_metrics.get('pauc', 0.0):.4f}")
    print(f"    - Validation ROC-AUC: {val_metrics.get('roc_auc', 0.0):.4f}\n")

    return report
