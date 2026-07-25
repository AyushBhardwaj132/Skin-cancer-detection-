from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize

from src.metrics import compute_pauc


def simple_average(predictions: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Compute standard arithmetic mean across prediction arrays.
    
    Args:
        predictions: List of 1D arrays or 2D array of shape (N_models, N_samples).
    """
    preds_arr = np.array(predictions)
    return np.mean(preds_arr, axis=0)


def rank_average(predictions: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Compute rank-averaged predictions across models.
    
    Converts probabilities to normalized percentile ranks (0.0 to 1.0) for each model,
    mitigating calibration shifts across different backbones.
    """
    preds_arr = np.array(predictions)
    ranked_preds = np.zeros_like(preds_arr, dtype=np.float64)
    
    for i in range(preds_arr.shape[0]):
        # Convert to 0-1 percentile ranks
        ranks = rankdata(preds_arr[i])
        ranked_preds[i] = (ranks - 1.0) / max(len(ranks) - 1, 1)
        
    return np.mean(ranked_preds, axis=0)


def weighted_average(
    predictions: list[np.ndarray] | np.ndarray,
    weights: list[float] | np.ndarray,
) -> np.ndarray:
    """Compute weighted average of prediction arrays."""
    preds_arr = np.array(predictions)
    weights_arr = np.array(weights, dtype=np.float64)
    weights_norm = weights_arr / np.sum(weights_arr)
    
    weighted_sum = np.zeros(preds_arr.shape[1], dtype=np.float64)
    for i in range(len(weights_norm)):
        weighted_sum += weights_norm[i] * preds_arr[i]
        
    return weighted_sum


def find_optimal_weights(
    predictions: list[np.ndarray],
    y_true: np.ndarray,
    max_fpr: float = 0.1,
) -> np.ndarray:
    """Find blend weights that maximize out-of-fold (OOF) pAUC score using SLSQP optimization."""
    preds_matrix = np.array(predictions)
    num_models = len(predictions)
    
    def loss_func(weights):
        blended = weighted_average(preds_matrix, weights)
        # Minimize negative pAUC
        pauc = compute_pauc(y_true, blended, max_fpr=max_fpr)
        return -pauc
        
    init_weights = np.ones(num_models) / num_models
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
    
    res = minimize(
        loss_func,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200},
    )
    
    opt_weights = res.x / np.sum(res.x)
    return opt_weights


def blend_predictions(
    predictions_dict: dict[str, np.ndarray],
    method: str = "rank",
    weights_dict: dict[str, float] | None = None,
) -> np.ndarray:
    """Blend dictionary of predictions (model_name -> prob_array).
    
    Supported methods: 'simple', 'weighted', 'rank'.
    """
    model_names = list(predictions_dict.keys())
    preds_list = [predictions_dict[name] for name in model_names]
    
    if method == "simple":
        return simple_average(preds_list)
    elif method == "rank":
        return rank_average(preds_list)
    elif method == "weighted":
        if weights_dict is None:
            weights = [1.0] * len(preds_list)
        else:
            weights = [weights_dict.get(name, 1.0) for name in model_names]
        return weighted_average(preds_list, weights)
    else:
        raise ValueError(f"Unknown ensembling method: {method}")
