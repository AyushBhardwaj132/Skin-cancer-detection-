from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize
from src.evaluation.metrics import compute_pauc


def simple_average(predictions_list: list[np.ndarray]) -> np.ndarray:
    """Mean probability predictions across multiple models."""
    return np.mean(predictions_list, axis=0)


def rank_average(predictions_list: list[np.ndarray]) -> np.ndarray:
    """Rank-averaged probabilities scaled to [0, 1]. Robust against calibration drift."""
    ranked = [rankdata(p) / len(p) for p in predictions_list]
    return np.mean(ranked, axis=0)


def weighted_average(predictions_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted average blend of predictions."""
    weights = np.array(weights) / np.sum(weights)
    weighted = np.zeros_like(predictions_list[0])
    for p, w in zip(predictions_list, weights):
        weighted += p * w
    return weighted


def find_optimal_weights(predictions_list: list[np.ndarray], y_true: np.ndarray, max_fpr: float = 0.1) -> np.ndarray:
    """Solves for optimal ensemble weights that maximize pAUC on validation set."""
    n_models = len(predictions_list)
    initial_weights = np.ones(n_models) / n_models

    def loss_func(weights):
        weights = np.array(weights)
        if np.sum(weights) <= 0:
            return 0.0
        blend = weighted_average(predictions_list, weights)
        return -compute_pauc(y_true, blend, max_fpr=max_fpr)

    bounds = [(0, 1) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})

    res = minimize(loss_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    best_w = res.x / np.sum(res.x)
    return best_w


def blend_predictions(
    predictions_list: list[np.ndarray],
    method: str = "rank",
    weights: list[float] | None = None,
) -> np.ndarray:
    if len(predictions_list) == 1:
        return predictions_list[0]

    method = method.lower()
    if method == "rank":
        return rank_average(predictions_list)
    elif method == "weighted" and weights is not None:
        return weighted_average(predictions_list, weights)
    else:
        return simple_average(predictions_list)
