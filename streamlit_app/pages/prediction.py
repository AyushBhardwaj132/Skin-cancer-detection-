"""Prediction Page view executing real model inference & Grad-CAM explainability."""
import streamlit as st
from PIL import Image

from streamlit_app.models.inference_engine import InferenceEngine
from streamlit_app.components.risk_gauge import render_risk_gauge
from streamlit_app.components.explainability import render_explainability_section


def render_prediction_page():
    """Renders real model inference results and explainability heatmaps."""
    if "uploaded_image" not in st.session_state or st.session_state["uploaded_image"] is None:
        st.warning("⚠️ No image payload found. Please upload a skin lesion image first.")
        if st.button("⬅️ Go to Upload Page", type="primary"):
            st.session_state["current_page"] = "upload"
            st.rerun()
        return

    image: Image.Image = st.session_state["uploaded_image"]
    metadata: dict = st.session_state.get("metadata_payload", {})

    with st.spinner("🔬 Running EfficientNetV2-S + Metadata Fusion Inference Engine..."):
        try:
            engine = InferenceEngine()
            if not engine.is_ready():
                st.error("❌ Model artifacts could not be loaded. Please ensure `best_model.pt` exists.")
                return

            # Execute model forward pass
            results = engine.predict(image, metadata, use_lesion_crop=True)
        except Exception as e:
            st.error(f"❌ Diagnostic inference error: {str(e)}")
            return

    st.success("✓ Diagnostic AI inference complete.")

    # Render Diagnostic Risk Gauge & Metric Cards
    render_risk_gauge(results)

    # Render Grad-CAM Explainability Heatmap Overlay
    render_explainability_section(results)

    # Action bar
    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Analyze Another Lesion", use_container_width=True):
            st.session_state["current_page"] = "upload"
            st.rerun()
    with c2:
        if st.button("📊 View Full Validation Metrics", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "results"
            st.rerun()
