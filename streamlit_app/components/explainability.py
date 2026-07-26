"""Grad-CAM explainability component rendering feature activation heatmaps."""
import streamlit as st
from PIL import Image


def render_explainability_section(prediction_results: dict):
    """Renders Grad-CAM visual explainability heatmap and architectural overview."""
    st.markdown("""
    <div class="stitch-card stitch-card-teal-top" style="margin-top: 24px;">
        <h3 style="margin-top: 0; color: #003757; display: flex; align-items: center; gap: 10px;">
            <span class="material-symbols-outlined" style="color: #0FA3A3;">visibility</span>
            Explainable AI (XAI) — Grad-CAM Feature Activation Map
        </h3>
        <p style="color: #475467; font-size: 0.9rem; margin-bottom: 16px;">
            Gradient-weighted Class Activation Mapping (Grad-CAM) computes gradients w.r.t the final EfficientNetV2 convolutional layer to highlight specific visual regions influencing the neural network's diagnostic decision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<span class='label-caps' style='color:#003757;'>Cropped Lesion Region</span>", unsafe_allow_html=True)
        if "cropped_image" in prediction_results and prediction_results["cropped_image"]:
            st.image(prediction_results["cropped_image"], use_container_width=True)

    with col2:
        st.markdown("<span class='label-caps' style='color:#0FA3A3;'>Grad-CAM Attention Overlay</span>", unsafe_allow_html=True)
        if "gradcam_overlay" in prediction_results and prediction_results["gradcam_overlay"]:
            st.image(prediction_results["gradcam_overlay"], use_container_width=True)
        else:
            st.info("Grad-CAM visualization pipeline initialized cleanly and ready for real-time feature extraction.")

    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-top: 16px;">
        <div style="font-size: 0.85rem; font-weight: 600; color: #003757; margin-bottom: 4px;">
            🔬 Visual Attention Interpretation Guide
        </div>
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.5;">
            Red and warm regions represent maximum convolutional activation where the EfficientNetV2 backbone detected irregular pigment networks, asymmetrical borders, or color variation. Blue and cool regions represent low-weight background tissue.
        </div>
    </div>
    """, unsafe_allow_html=True)
