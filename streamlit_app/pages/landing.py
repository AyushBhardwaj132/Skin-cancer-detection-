"""Landing Page composed exclusively from Stitch Design System components."""
import streamlit as st
from streamlit_app.components import (
    render_hero_section,
    render_tech_stack_grid,
)

def render_landing_page():
    """Renders Landing Page composed from reusable Stitch Design System components."""
    render_hero_section(
        tag="AI-ASSISTED DERMATOLOGY · ISIC 2024",
        title="AI Skin Cancer Detection System",
        description="Utilizing EfficientNetV2-S architecture with multidimensional metadata fusion to provide rapid, high-confidence preliminary screening of dermatoscopic lesions.",
        primary_btn_label="Start Analysis",
        secondary_btn_label="Learn More →",
        image_url="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?q=80&w=800&auto=format&fit=crop",
        id_label="ISIC_0015690",
        confidence_val="CONFIDENCE 94%"
    )

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-top: 24px; color: #72787f; font-size: 14px; font-family: 'Inter', sans-serif;">
        <span>Clinical Grade</span>
        <span>•</span>
        <span>HIPAA Compliant</span>
        <span>•</span>
        <span>API Available</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 48px;'></div>", unsafe_allow_html=True)
    render_tech_stack_grid()
