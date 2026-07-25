from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from src.metrics import compute_pauc


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> dict[str, float]:
    """Compute comprehensive evaluation metrics including ISIC pAUC."""
    valid = y_true >= 0
    y_true = y_true[valid]
    y_score = y_score[valid]
    
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": 0.0, "pauc": 0.0, "ap": 0.0, "best_threshold": 0.5, "best_f1": 0.0}
        
    roc_auc = roc_auc_score(y_true, y_score)
    pauc = compute_pauc(y_true, y_score, max_fpr=max_fpr)
    ap = average_precision_score(y_true, y_score)
    
    # Calculate best threshold by F1-score
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / np.maximum(precision + recall, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    
    return {
        "roc_auc": float(roc_auc),
        "pauc": float(pauc),
        "ap": float(ap),
        "best_threshold": best_thresh,
        "best_f1": best_f1,
    }


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str | Path,
    title_suffix: str = "",
    max_fpr: float = 0.1,
) -> None:
    """Plot and save ROC Curve and Precision-Recall Curve side-by-side."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid = y_true >= 0
    y_true = y_true[valid]
    y_score = y_score[valid]
    
    if len(np.unique(y_true)) < 2:
        print("Cannot plot ROC/PR curves with < 2 unique classes.")
        return
        
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    metrics = evaluate_predictions(y_true, y_score, max_fpr=max_fpr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color="#2980B9", lw=2, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    # Highlight pAUC region (FPR in [0, max_fpr])
    mask = fpr <= max_fpr
    ax1.fill_between(fpr[mask], tpr[mask], alpha=0.3, color="#27AE60", label=f"pAUC (FPR≤{max_fpr}) = {metrics['pauc']:.4f}")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_title(f"ROC Curve {title_suffix}")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color="#8E44AD", lw=2, label=f"PR-AUC / AP = {metrics['ap']:.4f}")
    ax2.axhline(y=np.mean(y_true), color="gray", linestyle="--", lw=1, label=f"Baseline ({np.mean(y_true):.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve {title_suffix}")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC/PR curves to {save_path}")
