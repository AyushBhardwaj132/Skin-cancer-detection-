from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_fold_dataframes(
    metadata_path: str | Path,
    fold_idx: int = 0,
    n_splits: int = 5,
    patient_col: str = "patient_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits metadata into train/validation sets using GroupKFold grouped by patient_id to prevent data leakage."""
    df = pd.read_csv(metadata_path)

    if patient_col not in df.columns:
        df[patient_col] = [f"P_{i}" for i in range(len(df))]

    gkf = GroupKFold(n_splits=n_splits)
    groups = df[patient_col]

    folds = list(gkf.split(df, groups=groups))
    train_idx, val_idx = folds[fold_idx]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df
