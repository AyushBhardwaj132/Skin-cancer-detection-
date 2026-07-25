from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def perform_error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    save_fig_path: str | Path | None = None,
) -> pd.DataFrame:
    """Analyze model errors (False Positives and False Negatives) by metadata subgroups.
    
    Args:
        df: DataFrame containing metadata columns (age_approx, anatom_site_general, etc.).
        y_true: Ground truth binary targets.
        y_score: Model predicted probabilities.
        threshold: Classification decision threshold.
        save_fig_path: Optional output path to save error breakdown visualization.
        
    Returns:
        DataFrame enriched with error classification tags (TP, TN, FP, FN).
    """
    analysis_df = df.copy().reset_index(drop=True)
    y_pred = (y_score >= threshold).astype(int)
    
    # Categorize prediction outcomes
    conditions = [
        (y_true == 1) & (y_pred == 1),
        (y_true == 0) & (y_pred == 0),
        (y_true == 0) & (y_pred == 1),
        (y_true == 1) & (y_pred == 0),
    ]
    choices = ['TP', 'TN', 'FP', 'FN']
    analysis_df['outcome'] = np.select(conditions, choices, default='Unknown')
    analysis_df['target'] = y_true
    analysis_df['pred_prob'] = y_score
    analysis_df['pred_binary'] = y_pred
    
    n_fp = (analysis_df['outcome'] == 'FP').sum()
    n_fn = (analysis_df['outcome'] == 'FN').sum()
    print(f"Error Analysis Summary (Threshold={threshold:.3f}):")
    print(f"  False Positives (FP): {n_fp}")
    print(f"  False Negatives (FN): {n_fn}")
    
    # Age binning
    if 'age_approx' in analysis_df.columns:
        analysis_df['age_group'] = pd.cut(
            analysis_df['age_approx'].fillna(-1),
            bins=[-2, 0, 30, 50, 70, 120],
            labels=['Missing', '<30', '30-50', '50-70', '70+']
        )
        
    # Plot breakdown if save_fig_path is provided
    if save_fig_path:
        save_fig_path = Path(save_fig_path)
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Breakdown by Age Group
        if 'age_group' in analysis_df.columns:
            age_errors = analysis_df[analysis_df['outcome'].isin(['FP', 'FN'])].groupby(['age_group', 'outcome']).size().unstack(fill_value=0)
            age_errors.plot(kind='bar', ax=axes[0], color=['#E74C3C', '#F39C12'], stacked=True)
            axes[0].set_title('Errors by Age Group', fontsize=12)
            axes[0].set_xlabel('Age Group')
            axes[0].set_ylabel('Error Count')
            axes[0].grid(True, alpha=0.3)
            
        # Breakdown by Anatomical Site
        if 'anatom_site_general' in analysis_df.columns:
            site_series = analysis_df['anatom_site_general'].fillna('Unknown').astype(str)
            site_df = analysis_df.assign(site=site_series)
            site_errors = site_df[site_df['outcome'].isin(['FP', 'FN'])].groupby(['site', 'outcome']).size().unstack(fill_value=0)
            site_errors.plot(kind='bar', ax=axes[1], color=['#E74C3C', '#F39C12'], stacked=True)
            axes[1].set_title('Errors by Anatomical Location', fontsize=12)
            axes[1].set_xlabel('Anatomical Location')
            axes[1].set_ylabel('Error Count')
            axes[1].tick_params(axis='x', rotation=30)
            axes[1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_fig_path, dpi=150)
        plt.close()
        print(f"Saved error analysis visualization to {save_fig_path}")
        
    return analysis_df
