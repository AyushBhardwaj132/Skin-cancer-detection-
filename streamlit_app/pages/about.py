"""About Page view detailing development journey & experimental milestones matching Stitch layout."""
import streamlit as st

def render_about_page():
    """Renders Development Journey and Milestone timeline."""
    st.markdown("""
    <div class="stitch-card stitch-card-primary-top">
        <h2 style="margin-top: 0; color: #003757;">Development Journey & Engineering Evolution</h2>
        <p style="color: #475467; font-size: 0.95rem;">
            From raw ISIC 2024 dataset exploration to production clinical deployment: a multi-phase research and engineering process.
        </p>
    </div>
    """, unsafe_allow_html=True)

    journey_steps = [
        {
            "phase": "Phase 1 • Quality Audit",
            "title": "Image Quality & Metadata Audit",
            "icon": "fact_check",
            "detail": "Analyzed over 400,000 whole-body photography images, identifying aspect ratio variations, missing metadata fields, and background artifacts. Built MetadataProcessor with StandardScaler & OneHotEncoder."
        },
        {
            "phase": "Phase 2 • Preprocessing",
            "title": "Preprocessing & Augmentation Experiments",
            "icon": "auto_fix_high",
            "detail": "Evaluated automated hair removal filter algorithms, DullRazor, and Albumentations transformations (RandomRotate90, ShiftScaleRotate, ColorJitter) to enhance model robustness across diverse skin tones."
        },
        {
            "phase": "Phase 3 • Resolution",
            "title": "224x224 vs 384x384 Resolution Experiment",
            "icon": "aspect_ratio",
            "detail": "Benchmarked spatial resolution impact. Increasing image resolution from 224x224 to 384x384 yielded a +3.8% boost in ROC-AUC by preserving fine structural pigment patterns."
        },
        {
            "phase": "Phase 4 • Lesion Crop",
            "title": "Lesion Center Crop ROI Experiment",
            "icon": "crop_free",
            "detail": "Developed OpenCV contour-based lesion localization with Otsu thresholding. Cropping square regions with a 20% margin eliminated background noise and accelerated model convergence."
        },
        {
            "phase": "Phase 5 • GPU Training",
            "title": "Kaggle GPU Acceleration & Loss Tuning",
            "icon": "speed",
            "detail": "Trained EfficientNetV2-S with Metadata Fusion on Kaggle P100/T4 GPUs using Focal Loss, AdamW optimizer, cosine annealing schedule, and Exponential Moving Average (EMA)."
        },
        {
            "phase": "Phase 6 • Production",
            "title": "Production Deployment & Stitch Integration",
            "icon": "rocket_launch",
            "detail": "Converted Stitch UI components into a modular Streamlit architecture with single-pass memory caching (@st.cache_resource), dynamic metric loading, and Grad-CAM explainability."
        },
    ]

    for step in journey_steps:
        st.markdown(f"""
        <div class="stitch-card" style="margin-bottom: 16px;">
            <div style="display: flex; gap: 16px; align-items: flex-start;">
                <div style="background: #003757; color: white; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <span class="material-symbols-outlined">{step['icon']}</span>
                </div>
                <div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #0FA3A3; text-transform: uppercase; letter-spacing: 0.05em;">
                        {step['phase']}
                    </div>
                    <h4 style="margin: 2px 0 6px 0; color: #003757; font-size: 1.1rem;">{step['title']}</h4>
                    <p style="margin: 0; font-size: 0.88rem; color: #475467; line-height: 1.5; opacity: 0.9;">
                        {step['detail']}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
