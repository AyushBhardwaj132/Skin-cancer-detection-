from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score

from src.utils import ensure_dir


def generate_evaluation_artifacts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    fold_idx: int = 0,
    threshold: float = 0.5,
    df_val: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Auto-generates ROC curve, PR curve, Confusion Matrix, and Hardest FP/FN tables."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    ensure_dir(figures_dir)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    artifact_paths = {}

    # 1. ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
        plt.axvspan(0, 0.1, alpha=0.15, color="red", label="pAUC Region (FPR <= 0.10)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — Fold {fold_idx}", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = figures_dir / f"roc_curve_fold{fold_idx}.png"
        plt.savefig(roc_path, dpi=300)
        plt.close()
        artifact_paths["roc_curve"] = roc_path
    except Exception as e:
        print(f"[WARN] Failed to generate ROC curve: {e}")

    # 2. Precision-Recall Curve
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

        plt.figure(figsize=(7, 6))
        plt.plot(rec, prec, color="#2ca02c", lw=2.5, label=f"PR Curve (AP = {ap_score:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — Fold {fold_idx}", fontsize=12, fontweight="bold")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = figures_dir / f"pr_curve_fold{fold_idx}.png"
        plt.savefig(pr_path, dpi=300)
        plt.close()
        artifact_paths["pr_curve"] = pr_path
    except Exception as e:
        print(f"[WARN] Failed to generate PR curve: {e}")

    # 3. Confusion Matrix
    try:
        preds = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (t={threshold:.2f}) — Fold {fold_idx}", fontsize=12, fontweight="bold")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Benign (0)", "Malignant (1)"])
        plt.yticks(tick_marks, ["Benign (0)", "Malignant (1)"])

        thresh_val = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh_val else "black",
                         fontsize=14, fontweight="bold")

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = figures_dir / f"confusion_matrix_fold{fold_idx}.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        artifact_paths["confusion_matrix"] = cm_path
    except Exception as e:
        print(f"[WARN] Failed to generate confusion matrix: {e}")

    # 4. Hardest False Positives & Hardest False Negatives
    if df_val is not None and len(df_val) == len(y_true):
        try:
            df_analysis = df_val.copy()
            df_analysis["target_true"] = y_true
            df_analysis["pred_prob"] = y_prob

            # Hardest False Positives: true label 0, highest predicted probability
            fp_df = df_analysis[df_analysis["target_true"] == 0].sort_values(by="pred_prob", ascending=False).head(10)
            fp_path = figures_dir / f"hardest_false_positives_fold{fold_idx}.csv"
            fp_df.to_csv(fp_path, index=False)
            artifact_paths["hardest_fp"] = fp_path

            # Hardest False Negatives: true label 1, lowest predicted probability
            fn_df = df_analysis[df_analysis["target_true"] == 1].sort_values(by="pred_prob", ascending=True).head(10)
            fn_path = figures_dir / f"hardest_false_negatives_fold{fold_idx}.csv"
            fn_df.to_csv(fn_path, index=False)
            artifact_paths["hardest_fn"] = fn_path
        except Exception as e:
            print(f"[WARN] Failed to export error analysis CSVs: {e}")

    return artifact_paths
