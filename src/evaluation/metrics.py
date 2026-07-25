from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


def compute_pauc(y_true: np.ndarray, y_pred: np.ndarray, max_fpr: float = 0.1) -> float:
    """Official ISIC 2024 competition metric: Partial Area Under ROC Curve (pAUC) above TPR threshold at max_fpr=0.1."""
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


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, max_fpr: float = 0.1) -> dict[str, float]:
    """Computes comprehensive evaluation suite: AUC, pAUC@0.1, Average Precision (AP)."""
    pauc = compute_pauc(y_true, y_pred, max_fpr=max_fpr)
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except Exception:
        auc = 0.0

    try:
        ap = float(average_precision_score(y_true, y_pred))
    except Exception:
        ap = 0.0

    return {
        "pauc": pauc,
        "auc": auc,
        "ap": ap,
    }
