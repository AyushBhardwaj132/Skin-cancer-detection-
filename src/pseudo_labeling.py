from __future__ import annotations

import numpy as np
import pandas as pd


def generate_pseudo_labels(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    pos_thresh: float = 0.99,
    neg_thresh: float = 0.01,
    image_id_col: str = "isic_id",
    target_col: str = "target",
) -> pd.DataFrame:
    """Generate high-confidence pseudo-labels from test predictions.
    
    Args:
        test_df: DataFrame containing test samples and metadata.
        predictions: Array of ensemble predicted probabilities for test samples.
        pos_thresh: Confidence threshold for positive class (>= pos_thresh -> 1.0).
        neg_thresh: Confidence threshold for negative class (<= neg_thresh -> 0.0).
        
    Returns:
        Filtered DataFrame containing only high-confidence pseudo-labeled samples.
    """
    df = test_df.copy().reset_index(drop=True)
    df["ensemble_prob"] = predictions
    
    pos_mask = df["ensemble_prob"] >= pos_thresh
    neg_mask = df["ensemble_prob"] <= neg_thresh
    
    confident_mask = pos_mask | neg_mask
    pseudo_df = df[confident_mask].copy()
    
    # Assign pseudo labels
    pseudo_df[target_col] = np.where(pseudo_df["ensemble_prob"] >= pos_thresh, 1.0, 0.0)
    
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    print(f"Generated {len(pseudo_df)} pseudo-labels ({n_pos} positive >= {pos_thresh}, {n_neg} negative <= {neg_thresh})")
    
    return pseudo_df


def merge_pseudo_labels(train_df: pd.DataFrame, pseudo_df: pd.DataFrame) -> pd.DataFrame:
    """Combine training DataFrame with pseudo-labeled test samples."""
    if pseudo_df.empty:
        print("No pseudo-labels to merge.")
        return train_df.copy()
        
    merged = pd.concat([train_df, pseudo_df], ignore_index=True)
    print(f"Merged train dataset shape: {train_df.shape[0]} -> {merged.shape[0]} samples")
    return merged
