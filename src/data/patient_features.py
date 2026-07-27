from __future__ import annotations

import numpy as np
import pandas as pd


def compute_patient_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Computes patient-level aggregations (mean lesion size, count, color variance)."""
    df_copy = df.copy()

    if "patient_id" not in df_copy.columns:
        df_copy["patient_id"] = "P_UNKNOWN"

    # Patient lesion counts
    patient_counts = df_copy.groupby("patient_id").size().reset_index(name="patient_lesion_count")
    df_copy = df_copy.merge(patient_counts, on="patient_id", how="left")

    agg_cols = {}
    if "clin_size_long_diam_mm" in df_copy.columns:
        agg_cols["clin_size_long_diam_mm"] = ["mean", "std", "max"]
    if "tbp_lv_areaMM2" in df_copy.columns:
        agg_cols["tbp_lv_areaMM2"] = ["mean", "std"]
    if "tbp_lv_color_std_mean" in df_copy.columns:
        agg_cols["tbp_lv_color_std_mean"] = ["mean"]
    if "tbp_lv_norm_color" in df_copy.columns:
        agg_cols["tbp_lv_norm_color"] = ["mean", "std"]

    if agg_cols:
        patient_aggs = df_copy.groupby("patient_id").agg(agg_cols)
        patient_aggs.columns = [f"patient_{col}_{stat}" for col, stat in patient_aggs.columns]
        df_copy = df_copy.merge(patient_aggs, on="patient_id", how="left")

    return df_copy


def compute_3d_spatial_and_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 3D spatial position norm and ratio features."""
    df_copy = df.copy()

    # 3D spatial Euclidean distance from scanner origin
    if all(col in df_copy.columns for col in ["tbp_lv_x", "tbp_lv_y", "tbp_lv_z"]):
        x = df_copy["tbp_lv_x"].fillna(0.0)
        y = df_copy["tbp_lv_y"].fillna(0.0)
        z = df_copy["tbp_lv_z"].fillna(0.0)
        df_copy["spatial_3d_distance"] = np.sqrt(x**2 + y**2 + z**2)

    # Color std to area ratio
    if "tbp_lv_color_std_mean" in df_copy.columns and "tbp_lv_areaMM2" in df_copy.columns:
        color = df_copy["tbp_lv_color_std_mean"].fillna(0.0)
        area = df_copy["tbp_lv_areaMM2"].fillna(1.0).replace(0, 1.0)
        df_copy["color_to_area_ratio"] = color / area

    return df_copy


def compute_ugly_duckling_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Ugly Duckling score comparing each lesion against patient baseline."""
    df_copy = df.copy()

    if "clin_size_long_diam_mm" in df_copy.columns and "patient_clin_size_long_diam_mm_mean" in df_copy.columns:
        std_col = df_copy.get("patient_clin_size_long_diam_mm_std")
        if std_col is not None:
            std = std_col.fillna(1.0).replace(0, 1.0)
        else:
            std = 1.0

        diff = df_copy["clin_size_long_diam_mm"] - df_copy["patient_clin_size_long_diam_mm_mean"]
        df_copy["ugly_duckling_score"] = (diff / std).fillna(0.0)
    else:
        df_copy["ugly_duckling_score"] = 0.0

    return df_copy


def enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Applies complete patient aggregation, 3D spatial distance, and Ugly Duckling feature engineering."""
    df_enriched = compute_patient_aggregates(df)
    df_enriched = compute_3d_spatial_and_ratio_features(df_enriched)
    df_enriched = compute_ugly_duckling_score(df_enriched)
    return df_enriched
