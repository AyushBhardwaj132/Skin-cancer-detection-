import numpy as np
from src.evaluation.metrics import compute_pauc, compute_all_metrics


def test_compute_pauc_perfect():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.85, 0.95, 0.88])
    score = compute_pauc(y_true, y_pred, max_fpr=0.1)
    assert score >= 0.0


def test_compute_all_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.85, 0.95, 0.88])
    metrics = compute_all_metrics(y_true, y_pred)
    assert "pauc" in metrics
    assert "auc" in metrics
    assert "ap" in metrics
