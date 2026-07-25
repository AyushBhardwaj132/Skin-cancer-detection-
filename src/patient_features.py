from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_DUCKLING_FEATURES = [
    'tbp_lv_A', 'tbp_lv_B', 'tbp_lv_C', 'tbp_lv_H', 'tbp_lv_L',
    'tbp_lv_areaMM2', 'tbp_lv_eccentricity', 'tbp_lv_color_std_mean',
    'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_symm_2axis',
]

def compute_patient_features(df: pd.DataFrame, patient_col='patient_id') -> pd.DataFrame:
    if patient_col not in df.columns:
        return pd.DataFrame()
        
    # Group by patient
    gb = df.groupby(patient_col)
    
    # Calculate features
    features = pd.DataFrame(index=gb.indices.keys())
    features.index.name = patient_col
    
    features['n_lesions'] = gb.size()
    
    if 'clin_size_long_diam_mm' in df.columns:
        features['mean_lesion_size'] = gb['clin_size_long_diam_mm'].mean()
        features['max_lesion_size'] = gb['clin_size_long_diam_mm'].max()
        features['std_lesion_size'] = gb['clin_size_long_diam_mm'].std().fillna(0.0)
    else:
        features['mean_lesion_size'] = np.nan
        features['max_lesion_size'] = np.nan
        features['std_lesion_size'] = 0.0
        
    if 'tbp_lv_areaMM2' in df.columns:
        features['mean_tbp_lv_areaMM2'] = gb['tbp_lv_areaMM2'].mean()
        features['max_tbp_lv_areaMM2'] = gb['tbp_lv_areaMM2'].max()
    else:
        features['mean_tbp_lv_areaMM2'] = np.nan
        features['max_tbp_lv_areaMM2'] = np.nan
        
    if 'tbp_lv_L' in df.columns:
        features['mean_tbp_lv_L'] = gb['tbp_lv_L'].mean()
        features['std_tbp_lv_L'] = gb['tbp_lv_L'].std().fillna(0.0)
    else:
        features['mean_tbp_lv_L'] = np.nan
        features['std_tbp_lv_L'] = 0.0
        
    if 'tbp_lv_color_std_mean' in df.columns:
        features['mean_tbp_lv_color_std_mean'] = gb['tbp_lv_color_std_mean'].mean()
    else:
        features['mean_tbp_lv_color_std_mean'] = np.nan
        
    if 'tbp_lv_eccentricity' in df.columns:
        features['mean_tbp_lv_eccentricity'] = gb['tbp_lv_eccentricity'].mean()
    else:
        features['mean_tbp_lv_eccentricity'] = np.nan
        
    if 'tbp_lv_nevi_confidence' in df.columns:
        features['mean_tbp_lv_nevi_confidence'] = gb['tbp_lv_nevi_confidence'].mean()
    else:
        features['mean_tbp_lv_nevi_confidence'] = np.nan
        
    return features.reset_index()

def compute_ugly_duckling_score(df: pd.DataFrame, feature_cols=None, patient_col='patient_id') -> np.ndarray:
    if feature_cols is None:
        feature_cols = DEFAULT_DUCKLING_FEATURES
        
    if patient_col not in df.columns or len(df) == 0:
        return np.zeros(len(df), dtype=np.float32)
        
    # Get available features
    available_cols = [col for col in feature_cols if col in df.columns]
    if not available_cols:
        return np.zeros(len(df), dtype=np.float32)
        
    scores = np.zeros(len(df), dtype=np.float32)
    
    # Fill NaN with 0 for computation
    data = df[available_cols].fillna(0.0).values
    
    # Map patient to index for fast lookup
    patient_ids = df[patient_col].values
    
    # For each patient, compute distance to mean
    unique_patients = np.unique(patient_ids)
    
    for pid in unique_patients:
        mask = (patient_ids == pid)
        patient_data = data[mask]
        
        if len(patient_data) <= 1:
            # 1 lesion -> distance is 0
            scores[mask] = 0.0
        else:
            mean_vec = np.mean(patient_data, axis=0)
            # Euclidean distance
            dist = np.sqrt(np.sum((patient_data - mean_vec) ** 2, axis=1))
            scores[mask] = dist
            
    return scores

def enrich_metadata(df: pd.DataFrame, patient_col='patient_id') -> pd.DataFrame:
    df = df.copy()
    
    if patient_col not in df.columns:
        return df
        
    # Compute patient features
    pat_features = compute_patient_features(df, patient_col=patient_col)
    
    # Merge
    if not pat_features.empty:
        df = df.merge(pat_features, on=patient_col, how='left')
        
    # Compute ugly duckling
    ud_score = compute_ugly_duckling_score(df, feature_cols=DEFAULT_DUCKLING_FEATURES, patient_col=patient_col)
    df['ugly_duckling_score'] = ud_score
    
    return df
