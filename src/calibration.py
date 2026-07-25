from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    valid = y_true >= 0
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)


class TemperatureScaler:
    """Post-hoc probability calibration using Temperature Scaling on logits."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray):
        """Find optimal temperature T > 0 minimizing NLL loss on validation logits."""
        def nll_loss(t):
            temp = t[0]
            scaled_logits = logits / max(temp, 1e-4)
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
            return loss

        res = minimize(nll_loss, [1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
        self.temperature = float(res.x[0])
        print(f"Optimal Temperature T = {self.temperature:.4f}")
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled_logits = logits / max(self.temperature, 1e-4)
        return 1.0 / (1.0 + np.exp(-scaled_logits))


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_uncal: np.ndarray,
    y_prob_cal: np.ndarray,
    save_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Plot and save calibration reliability diagram comparing before and after calibration."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    ece_uncal = compute_ece(y_true, y_prob_uncal, n_bins=n_bins)
    ece_cal = compute_ece(y_true, y_prob_cal, n_bins=n_bins)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    acc_uncal = []
    acc_cal = []
    
    for i in range(n_bins):
        mask_uncal = (y_prob_uncal >= bin_boundaries[i]) & (y_prob_uncal < bin_boundaries[i+1])
        mask_cal = (y_prob_cal >= bin_boundaries[i]) & (y_prob_cal < bin_boundaries[i+1])
        
        acc_uncal.append(np.mean(y_true[mask_uncal]) if np.sum(mask_uncal) > 0 else np.nan)
        acc_cal.append(np.mean(y_true[mask_cal]) if np.sum(mask_cal) > 0 else np.nan)
        
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', lw=1)
    ax.plot(bin_centers, acc_uncal, 'o-', color='#E74C3C', label=f'Uncalibrated (ECE={ece_uncal:.4f})')
    ax.plot(bin_centers, acc_cal, 's-', color='#27AE60', label=f'Calibrated (ECE={ece_cal:.4f})')
    
    ax.set_xlabel('Mean Predicted Probability (Confidence)')
    ax.set_ylabel('Fraction of Positives (Accuracy)')
    ax.set_title('Probability Calibration Reliability Diagram')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved calibration curve to {save_path}")
