from __future__ import annotations

import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

METADATA_NUMERIC_COLS = [
    "age_approx", "clin_size_long_diam_mm", "tbp_lv_A", "tbp_lv_Aext", "tbp_lv_B", "tbp_lv_Bext",
    "tbp_lv_C", "tbp_lv_Cext", "tbp_lv_H", "tbp_lv_Hext", "tbp_lv_L", "tbp_lv_Lext",
    "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean", "tbp_lv_deltaA",
    "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm", "tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM", "tbp_lv_nevi_confidence", "tbp_lv_norm_border", "tbp_lv_norm_color",
    "tbp_lv_perimeterMM", "tbp_lv_radial_color_std_max", "tbp_lv_stdL", "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle", "tbp_lv_x", "tbp_lv_y", "tbp_lv_z",
]

METADATA_CATEGORICAL_COLS = [
    "sex", "anatom_site_general", "image_type", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple",
]


class MetadataProcessor:
    """Encodes and normalizes tabular patient & lesion metadata into dense feature tensors."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.num_cols = list(METADATA_NUMERIC_COLS)
        self.cat_cols = list(METADATA_CATEGORICAL_COLS)
        self.fitted_num_cols = []
        self.fitted_cat_cols = []
        self.is_fitted = False

    def get_feature_dim(self) -> int:
        num_dim = len(self.scaler.mean_) if hasattr(self.scaler, "mean_") else len(self.num_cols)
        cat_dim = sum(len(c) for c in self.encoder.categories_) if hasattr(self.encoder, "categories_") else 10
        return num_dim + cat_dim

    def fit(self, df: pd.DataFrame) -> MetadataProcessor:
        df_copy = df.copy()
        existing_num = [c for c in self.num_cols if c in df_copy.columns]
        if existing_num:
            self.fitted_num_cols = existing_num
            num_data = df_copy[existing_num].fillna(df_copy[existing_num].median()).fillna(0.0)
            self.scaler.fit(num_data)

        existing_cat = [c for c in self.cat_cols if c in df_copy.columns]
        if existing_cat:
            self.fitted_cat_cols = existing_cat
            cat_data = df_copy[existing_cat].fillna("Unknown").astype(str)
            self.encoder.fit(cat_data)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df_copy = df.copy()

        fitted_num = getattr(self, "fitted_num_cols", [c for c in self.num_cols if c in df_copy.columns])
        if fitted_num and hasattr(self.scaler, "mean_"):
            for c in fitted_num:
                if c not in df_copy.columns:
                    df_copy[c] = 0.0
            num_data = df_copy[fitted_num].fillna(0.0)
            num_features = self.scaler.transform(num_data)
        else:
            num_features = np.zeros((len(df), len(self.num_cols)), dtype=np.float32)

        fitted_cat = getattr(self, "fitted_cat_cols", [c for c in self.cat_cols if c in df_copy.columns])
        if fitted_cat and hasattr(self.encoder, "categories_"):
            for c in fitted_cat:
                if c not in df_copy.columns:
                    df_copy[c] = "Unknown"
            cat_data = df_copy[fitted_cat].fillna("Unknown").astype(str)
            cat_features = self.encoder.transform(cat_data)
        else:
            cat_features = np.zeros((len(df), 10), dtype=np.float32)

        # Include engineered features if present (exclude ID/string columns)
        _exclude = {"patient_id", "patient_site"}
        extra_cols = [
            c for c in df_copy.columns
            if (c.startswith("patient_") or c.startswith("ugly_duckling"))
            and c not in _exclude
            and pd.api.types.is_numeric_dtype(df_copy[c])
        ]
        if extra_cols:
            extra_data = df_copy[extra_cols].fillna(0.0).values.astype(np.float32)
            return np.hstack([num_features, cat_features, extra_data]).astype(np.float32)

        return np.hstack([num_features, cat_features]).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> MetadataProcessor:
        obj: MetadataProcessor = joblib.load(path)
        # Backward compatibility: ensure essential attributes exist
        if not hasattr(obj, "num_cols"):
            obj.num_cols = list(METADATA_NUMERIC_COLS)
        if not hasattr(obj, "cat_cols"):
            obj.cat_cols = list(METADATA_CATEGORICAL_COLS)
        if not hasattr(obj, "scaler"):
            obj.scaler = StandardScaler()
        if not hasattr(obj, "encoder"):
            obj.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if not hasattr(obj, "fitted_num_cols"):
            obj.fitted_num_cols = [c for c in obj.num_cols if hasattr(obj.scaler, "mean_")]
        if not hasattr(obj, "fitted_cat_cols"):
            obj.fitted_cat_cols = [c for c in obj.cat_cols if hasattr(obj.encoder, "categories_")]
        if hasattr(obj.scaler, "mean_") or hasattr(obj.encoder, "categories_"):
            obj.is_fitted = True
        return obj
