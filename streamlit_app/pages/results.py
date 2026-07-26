"""Results Dashboard view displaying evaluation metrics & curves dynamically."""
import streamlit as st
from pathlib import Path
from PIL import Image

from streamlit_app.components.metrics_grid import render_metrics_grid
from streamlit_app.config.settings import EVALUATION_DIR, BASE_DIR

def render_results_page():
    """Renders the AI Results & Evaluation Dashboard."""
    render_metrics_grid()

    st.markdown("### 📈 Evaluation Diagnostic Curves & Confusion Matrix", unsafe_allow_html=True)
    st.caption("Generated during patient-aware GroupKFold validation evaluation:")

    eval_dir = EVALUATION_DIR
    if not eval_dir.exists():
        eval_dir = BASE_DIR / "outputs" / "evaluation" / "dev"

    c1, c2 = st.columns(2)

    roc_path = eval_dir / "roc_curve.png"
    pr_path = eval_dir / "precision_recall_curve.png"
    cm_path = eval_dir / "confusion_matrix.png"
    calib_path = eval_dir / "calibration_curve.png"

    with c1:
        if roc_path.exists():
            st.markdown("<span class='label-caps' style='color:#003757;'>ROC-AUC Curve</span>", unsafe_allow_html=True)
            st.image(str(roc_path), use_container_width=True)
        if cm_path.exists():
            st.markdown("<span class='label-caps' style='color:#003757;'>Confusion Matrix</span>", unsafe_allow_html=True)
            st.image(str(cm_path), use_container_width=True)

    with c2:
        if pr_path.exists():
            st.markdown("<span class='label-caps' style='color:#0FA3A3;'>Precision-Recall Curve</span>", unsafe_allow_html=True)
            st.image(str(pr_path), use_container_width=True)
        if calib_path.exists():
            st.markdown("<span class='label-caps' style='color:#0FA3A3;'>Calibration Curve</span>", unsafe_allow_html=True)
            st.image(str(calib_path), use_container_width=True)
