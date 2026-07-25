from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERICAL_FEATURES = [
    'age_approx', 'clin_size_long_diam_mm',
    'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext',
    'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2',
    'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL',
    'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence', 'tbp_lv_norm_border',
    'tbp_lv_norm_color', 'tbp_lv_perimeterMM',
    'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis',
    'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
]

CATEGORICAL_FEATURES = [
    'sex', 'anatom_site_general', 'tbp_lv_location_simple', 'tbp_tile_type',
]

class MetadataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.medians = {}
        self.categories = {}
        self.feature_dim = 0
        
    def fit(self, train_df: pd.DataFrame):
        # Learn medians for numerical
        for col in NUMERICAL_FEATURES:
            if col in train_df.columns:
                self.medians[col] = train_df[col].median()
                if pd.isna(self.medians[col]):
                    self.medians[col] = 0.0
            else:
                self.medians[col] = 0.0
                
        # Fill NA for scaling
        num_data = pd.DataFrame(index=train_df.index)
        for col in NUMERICAL_FEATURES:
            if col in train_df.columns:
                num_data[col] = train_df[col].fillna(self.medians[col])
            else:
                num_data[col] = self.medians[col]
                
        self.scaler.fit(num_data[NUMERICAL_FEATURES])
        
        # Learn categories
        for col in CATEGORICAL_FEATURES:
            if col in train_df.columns:
                # Fillna with a specific token and convert to string
                cats = train_df[col].fillna('Unknown').astype(str).unique().tolist()
            else:
                cats = ['Unknown']
            self.categories[col] = cats
            
        # Calculate feature dim
        # Numerical features count
        dim = len(NUMERICAL_FEATURES)
        # Categorical features count (one-hot encoded)
        for col in CATEGORICAL_FEATURES:
            dim += len(self.categories[col])
            
        self.feature_dim = dim
        return self
        
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Process numerical
        num_data = pd.DataFrame(index=df.index)
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                num_data[col] = df[col].fillna(self.medians[col])
            else:
                num_data[col] = self.medians[col]
                
        num_features = self.scaler.transform(num_data[NUMERICAL_FEATURES])
        
        # Process categorical
        cat_features_list = []
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                val = df[col].fillna('Unknown').astype(str)
            else:
                val = pd.Series(['Unknown'] * len(df), index=df.index)
                
            # Create one-hot
            cat_array = np.zeros((len(df), len(self.categories[col])), dtype=np.float32)
            for i, cat in enumerate(self.categories[col]):
                cat_array[:, i] = (val == cat).astype(np.float32)
            cat_features_list.append(cat_array)
            
        if cat_features_list:
            cat_features = np.hstack(cat_features_list)
            all_features = np.hstack([num_features, cat_features])
        else:
            all_features = num_features
            
        return all_features.astype(np.float32)
        
    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        self.fit(train_df)
        return self.transform(train_df)
        
    def get_feature_dim(self) -> int:
        return self.feature_dim
        
    def save(self, path: str):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> MetadataProcessor:
        return joblib.load(path)
