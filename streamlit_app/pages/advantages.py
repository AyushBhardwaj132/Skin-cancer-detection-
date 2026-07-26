"""Advantages Page view highlighting 8 core project differentiators matching Stitch layout."""
import streamlit as st
from streamlit_app.components.cards import render_feature_card

def render_advantages_page():
    """Renders the Why Our Project Is Different advantages page."""
    st.markdown("""
    <div class="stitch-card stitch-card-teal-top">
        <h2 style="margin-top: 0; color: #003757;">Why Our Model & Architecture Is Better</h2>
        <p style="color: #475467; font-size: 0.95rem;">
            DermaVision AI moves beyond basic image classification by introducing multimodal fusion, patient-aware validation, and production explainability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    advantages = [
        {
            "title": "Metadata Fusion",
            "tag": "Multimodal AI",
            "icon": "hub",
            "desc": "Combines deep visual features from EfficientNetV2 with 47 tabular demographic, spatial, and 3D body measurements via a parallel MLP embedding network."
        },
        {
            "title": "Lesion-Centered Preprocessing",
            "tag": "Computer Vision",
            "icon": "crop_center",
            "desc": "Automated OpenCV Otsu contour detection crops square region around lesion ROI with a 20% margin, stripping away healthy skin noise and clothing artifacts."
        },
        {
            "title": "Patient-Aware Validation",
            "tag": "Robust ML",
            "icon": "groups",
            "desc": "Uses GroupKFold split on patient_id so lesions from the same patient never leak across train and validation sets, ensuring realistic clinical generalization."
        },
        {
            "title": "Class Imbalance Handling",
            "tag": "Focal Loss",
            "icon": "scale",
            "desc": "Employs Focal Loss with modulating gamma factor to handle severe class imbalance (less than 1% malignant samples) without over-predicting false positives."
        },
        {
            "title": "GPU Training Pipeline",
            "tag": "Hardware Accelerated",
            "icon": "speed",
            "desc": "Fully optimized PyTorch training pipeline supporting Kaggle P100/T4 GPUs, mixed precision (AMP), gradient accumulation, and EMA checkpointing."
        },
        {
            "title": "Real Production Deployment",
            "tag": "Streamlit App",
            "icon": "rocket_launch",
            "desc": "Packaged into a high-performance Streamlit clinical dashboard loaded with single-pass memory caching (@st.cache_resource) and interactive UI components."
        },
        {
            "title": "Explainable AI Ready",
            "tag": "Grad-CAM",
            "icon": "visibility",
            "desc": "Generates gradient-weighted class activation heatmaps overlaying input images, allowing clinicians to visually inspect diagnostic focal points."
        },
        {
            "title": "Modular Architecture",
            "tag": "Clean Code",
            "icon": "widgets",
            "desc": "Refactored into clean, typed Python packages (config, models, components, utils, pages) with docstrings, logging, and strict separation of concerns."
        },
    ]

    cols = st.columns(2)
    for idx, adv in enumerate(advantages):
        col = cols[idx % 2]
        with col:
            render_feature_card(adv["title"], adv["desc"], adv["icon"], adv["tag"])
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
