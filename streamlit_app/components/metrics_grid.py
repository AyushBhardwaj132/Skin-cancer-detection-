"""Evaluation metrics grid component loading values dynamically matching Stitch design 1:1."""
import streamlit as st
from streamlit_app.utils.metrics_loader import load_evaluation_metrics

def render_metrics_grid():
    """Renders 8 evaluation metric boxes matching Stitch Model Performance Analytics 1:1."""
    metrics = load_evaluation_metrics()

    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 class="font-headline-lg" style="margin: 0; font-size: 32px; color: #003757;">
            Model Performance Analytics
        </h1>
        <p class="font-body-lg" style="margin-top: 6px; color: #42474e; font-size: 16px;">
            Patient-aware GroupKFold validation performance benchmark on the ISIC 2024 Dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom: 16px;">
            <div class="val">{metrics['roc_auc']:.4f}</div>
            <div class="lbl">ROC-AUC</div>
        </div>
        <div class="metric-box">
            <div class="val">{metrics['precision']:.4f}</div>
            <div class="lbl">Precision</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom: 16px;">
            <div class="val">{metrics['pauc']:.4f}</div>
            <div class="lbl">pAUC (TPR > 80%)</div>
        </div>
        <div class="metric-box">
            <div class="val">{metrics['recall']:.4f}</div>
            <div class="lbl">Recall (Sensitivity)</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom: 16px;">
            <div class="val">{metrics['accuracy']:.4f}</div>
            <div class="lbl">Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="val">{metrics['f1']:.4f}</div>
            <div class="lbl">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom: 16px;">
            <div class="val">{metrics['balanced_accuracy']:.4f}</div>
            <div class="lbl">Balanced Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="val">{metrics['mcc']:.4f}</div>
            <div class="lbl">MCC (Matthews Corr)</div>
        </div>
        """, unsafe_allow_html=True)
