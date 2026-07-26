"""Reusable card components matching Stitch design system."""
import streamlit as st

def render_tech_stack_grid():
    """Renders professional tech stack cards matching Stitch design."""
    tech_items = [
        {"name": "Python 3.10+", "role": "Core Language & Backend Ecosystem", "icon": "code", "color": "#3776AB"},
        {"name": "PyTorch", "role": "Deep Learning & Neural Network Pipeline", "icon": "dataset", "color": "#EE4C2C"},
        {"name": "EfficientNetV2-S", "role": "Fused CNN Visual Representation Backbone", "icon": "view_in_ar", "color": "#003757"},
        {"name": "Albumentations", "role": "Fast Image Data Augmentation Pipeline", "icon": "auto_fix_high", "color": "#0FA3A3"},
        {"name": "OpenCV", "role": "Computer Vision & Lesion Crop Preprocessing", "icon": "crop_free", "color": "#5C3EE8"},
        {"name": "NumPy & Pandas", "role": "Tabular Metadata Engineering & Vector Math", "icon": "table_chart", "color": "#130654"},
        {"name": "Scikit-Learn", "role": "StandardScaler & OneHotEncoder Transformation", "icon": "analytics", "color": "#F7931E"},
        {"name": "Streamlit", "role": "Production Interactive Medical Frontend UI", "icon": "dashboard", "color": "#FF4B4B"},
        {"name": "GitHub", "role": "Version Control & Modular Source Code Storage", "icon": "terminal", "color": "#24292E"},
        {"name": "Kaggle GPU P100/T4", "role": "Accelerated Deep Model Training Hardware", "icon": "speed", "color": "#20BEFF"},
    ]

    st.markdown("### 🛠️ Production Technology Stack", unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, tech in enumerate(tech_items):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
            <div class="stitch-card" style="padding: 16px; margin-bottom: 12px; display: flex; align-items: center; gap: 16px;">
                <div style="background: {tech['color']}15; color: {tech['color']}; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <span class="material-symbols-outlined">{tech['icon']}</span>
                </div>
                <div>
                    <div style="font-weight: 700; font-size: 1rem; color: #003757; font-family: 'Hanken Grotesk', sans-serif;">
                        {tech['name']}
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b;">
                        {tech['role']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_feature_card(title: str, description: str, icon: str, advantage_tag: str):
    """Renders single advantage feature card matching Stitch theme."""
    st.markdown(f"""
    <div class="stitch-card stitch-card-teal-top" style="height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <div style="background: #0FA3A315; color: #0FA3A3; width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                <span class="material-symbols-outlined">{icon}</span>
            </div>
            <span class="badge-low" style="font-size: 0.75rem;">✓ {advantage_tag}</span>
        </div>
        <h4 style="margin: 0 0 8px 0; font-size: 1.1rem; color: #003757;">{title}</h4>
        <p style="margin: 0; font-size: 0.88rem; color: #475467; line-height: 1.55;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
