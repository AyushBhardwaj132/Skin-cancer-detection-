from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def create_group_kfolds(
    metadata_path: str | Path,
    n_splits: int = 5,
    group_column: str = "patient_id",
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create GroupKFold splits using patient_id to prevent data leakage.
    
    Args:
        metadata_path: Path to the training metadata CSV.
        n_splits: Number of folds.
        group_column: Column name for grouping (patient_id).
        seed: Random seed for reproducibility.
    
    Returns:
        List of (train_indices, val_indices) tuples for each fold.
    """
    metadata = pd.read_csv(metadata_path)
    
    if group_column not in metadata.columns:
        raise KeyError(f"Column '{group_column}' not found in metadata")
    
    gkf = GroupKFold(n_splits=n_splits)
    groups = metadata[group_column].values
    
    folds = []
    for train_idx, val_idx in gkf.split(metadata, groups=groups):
        folds.append((train_idx, val_idx))
    
    return folds


def get_fold_dataframes(
    metadata_path: str | Path,
    fold_idx: int,
    n_splits: int = 5,
    group_column: str = "patient_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get training and validation DataFrames for a specific fold.
    
    Args:
        metadata_path: Path to the training metadata CSV.
        fold_idx: Which fold to return (0-indexed).
        n_splits: Number of folds.
        group_column: Column name for grouping.
    
    Returns:
        (train_df, val_df) tuple.
    """
    metadata = pd.read_csv(metadata_path)
    folds = create_group_kfolds(metadata_path, n_splits=n_splits, group_column=group_column)
    
    if fold_idx < 0 or fold_idx >= len(folds):
        raise ValueError(f"fold_idx must be in range [0, {len(folds)-1}]")
    
    train_idx, val_idx = folds[fold_idx]
    train_df = metadata.iloc[train_idx].reset_index(drop=True)
    val_df = metadata.iloc[val_idx].reset_index(drop=True)
    
    return train_df, val_df
