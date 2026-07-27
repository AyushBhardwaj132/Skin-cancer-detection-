from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)


def compute_pauc(y_true: np.ndarray, y_pred: np.ndarray, max_fpr: float = 0.1) -> float:
    """Official ISIC 2024 competition metric: Partial Area Under ROC Curve (pAUC) above TPR threshold at max_fpr=0.1.

    Root Cause Analysis for Fold 1 pAUC = 0.0000:
    The official formula calculates pAUC = trapz(max(0, TPR - 0.8), FPR) / (0.2 * 0.1).
    If the model's predictions in Fold 1 fail to rank positive melanoma samples in the top 10% FPR
    with True Positive Rate (TPR) exceeding 0.80, max(0, TPR - 0.8) equals 0 for all points in FPR <= 0.10,
    returning an exact 0.0000 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2:
        return 0.0

    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        if max_fpr is not None and max_fpr < 1.0:
            stop_idx = np.searchsorted(fpr, max_fpr, side="right")
            if stop_idx < len(fpr):
                fpr_sub = np.append(fpr[:stop_idx], max_fpr)
                tpr_sub = np.append(tpr[:stop_idx], np.interp(max_fpr, fpr, tpr))
            else:
                fpr_sub = fpr
                tpr_sub = tpr

            min_tpr = 0.8
            tpr_sub = np.maximum(tpr_sub - min_tpr, 0)
            pauc = np.trapz(tpr_sub, fpr_sub)
            max_possible = (1.0 - min_tpr) * max_fpr
            return float(pauc / max_possible) if max_possible > 0 else 0.0
        else:
            return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return 0.0


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    num_thresholds: int = 100,
) -> tuple[float, float]:
    """Grid search for decision threshold in [0.01, 0.99] maximizing F1 or Balanced Accuracy."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, preds, zero_division=0))
        elif metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, preds))
        elif metric == "mcc":
            score = float(matthews_corrcoef(y_true, preds))
        else:
            score = float(f1_score(y_true, preds, zero_division=0))

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, best_score


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, max_fpr: float = 0.1) -> dict[str, float]:
    """Computes comprehensive evaluation suite: pAUC@0.1, AUC, AP, optimal threshold, F1, Acc, Prec, Rec."""
    pauc = compute_pauc(y_true, y_pred, max_fpr=max_fpr)
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except Exception:
        auc = 0.0

    try:
        ap = float(average_precision_score(y_true, y_pred))
    except Exception:
        ap = 0.0

    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_pred, metric="f1")
    binary_preds = (y_pred >= opt_thresh).astype(int)

    return {
        "pauc": pauc,
        "auc": auc,
        "ap": ap,
        "optimal_threshold": opt_thresh,
        "f1_optimal": opt_f1,
        "accuracy_optimal": float(accuracy_score(y_true, binary_preds)),
        "precision_optimal": float(precision_score(y_true, binary_preds, zero_division=0)),
        "recall_optimal": float(recall_score(y_true, binary_preds, zero_division=0)),
    }
