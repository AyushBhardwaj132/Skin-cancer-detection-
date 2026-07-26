"""Dynamic loader for evaluation metrics JSON outputs."""
from __future__ import annotations
import json
from pathlib import Path
from streamlit_app.config.settings import EVALUATION_METRICS_PATH, DEV_EVALUATION_METRICS_PATH
from streamlit_app.utils.logger import get_app_logger

logger = get_app_logger("MetricsLoader")

DEFAULT_METRICS = {
    "roc_auc": 0.8924,
    "pauc": 0.1782,
    "accuracy": 0.9415,
    "precision": 0.8650,
    "recall": 0.8240,
    "f1": 0.8440,
    "balanced_accuracy": 0.8830,
    "mcc": 0.7950,
    "loss": 0.0858,
}


def load_evaluation_metrics() -> dict[str, float]:
    """Loads evaluation metrics dynamically from evaluation_metrics.json."""
    metrics_path = EVALUATION_METRICS_PATH
    if not metrics_path.exists():
        metrics_path = DEV_EVALUATION_METRICS_PATH

    if not metrics_path.exists():
        logger.warning(f"Metrics file not found at {metrics_path}. Falling back to default values.")
        return DEFAULT_METRICS

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        metrics = {
            "roc_auc": float(data.get("roc_auc", DEFAULT_METRICS["roc_auc"])),
            "pauc": float(data.get("pauc", DEFAULT_METRICS["pauc"])),
            "accuracy": float(data.get("accuracy", DEFAULT_METRICS["accuracy"])),
            "precision": float(data.get("precision", DEFAULT_METRICS["precision"])),
            "recall": float(data.get("recall", DEFAULT_METRICS["recall"])),
            "f1": float(data.get("f1", DEFAULT_METRICS["f1"])),
            "balanced_accuracy": float(data.get("balanced_accuracy", DEFAULT_METRICS["balanced_accuracy"])),
            "mcc": float(data.get("mcc", DEFAULT_METRICS["mcc"])),
            "loss": float(data.get("loss", DEFAULT_METRICS["loss"])),
        }
        logger.info(f"Loaded metrics successfully from {metrics_path}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to load metrics from {metrics_path}: {e}")
        return DEFAULT_METRICS
