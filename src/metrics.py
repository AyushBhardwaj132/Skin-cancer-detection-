from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score


def compute_pauc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> float:
    """
    Compute the partial Area Under the ROC Curve (pAUC).
    
    pAUC is the area under the ROC curve restricted to False Positive Rates <= max_fpr.
    This is the official metric used in the ISIC 2024 challenge.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_score: Target scores (probabilities or logits).
        max_fpr: Maximum false positive rate threshold (default 0.1 for ISIC).
    
    Returns:
        pAUC value normalized to [0, 1].
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # Find indices where fpr <= max_fpr
    valid_mask = fpr <= max_fpr
    fpr_clipped = fpr[valid_mask]
    tpr_clipped = tpr[valid_mask]
    
    # Compute the area under the clipped curve
    partial_auc = auc(fpr_clipped, tpr_clipped)
    
    # Normalize by the maximum possible pAUC
    # which is achieved when all positives are ranked before negatives at FPR <= max_fpr
    max_partial_auc = max_fpr
    if max_partial_auc > 0:
        normalized_pauc = partial_auc / max_partial_auc
    else:
        normalized_pauc = 0.0
    
    return normalized_pauc


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute standard ROC-AUC score."""
    return roc_auc_score(y_true, y_score)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> dict:
    """Compute pAUC, ROC-AUC, and other metrics."""
    return {
        "pauc": compute_pauc(y_true, y_score, max_fpr=max_fpr),
        "roc_auc": compute_roc_auc(y_true, y_score),
    }
