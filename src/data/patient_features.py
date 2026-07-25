from __future__ import annotations

import pandas as pd


def compute_patient_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Computes patient-level aggregations (mean lesion size, count, color variance)."""
    df_copy = df.copy()

    if "patient_id" not in df_copy.columns:
        df_copy["patient_id"] = "P_UNKNOWN"

    agg_cols = {}
    if "clin_size_long_diam_mm" in df_copy.columns:
        agg_cols["clin_size_long_diam_mm"] = ["mean", "std", "max"]
    if "tbp_lv_areaMM2" in df_copy.columns:
        agg_cols["tbp_lv_areaMM2"] = ["mean", "std"]
    if "tbp_lv_color_std_mean" in df_copy.columns:
        agg_cols["tbp_lv_color_std_mean"] = ["mean"]

    if not agg_cols:
        return df_copy

    patient_aggs = df_copy.groupby("patient_id").agg(agg_cols)
    patient_aggs.columns = [f"patient_{col}_{stat}" for col, stat in patient_aggs.columns]

    df_copy = df_copy.merge(patient_aggs, on="patient_id", how="left")
    return df_copy


def compute_ugly_duckling_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Ugly Duckling score comparing each lesion against the patient's mean lesion characteristics."""
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
    """Applies patient aggregation and Ugly Duckling feature engineering."""
    df_enriched = compute_patient_aggregates(df)
    df_enriched = compute_ugly_duckling_score(df_enriched)
    return df_enriched
