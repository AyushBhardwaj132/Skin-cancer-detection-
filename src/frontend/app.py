from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.config import Config
from src.models.fusion_model import FusionModel
from src.data.metadata import MetadataProcessor
from src.data.transforms import build_transforms
from src.utils import get_device, load_checkpoint
from src.utils.xai import GradCAM, overlay_heatmap_on_image

@st.cache_resource
def load_cached_artifacts():
    config = Config()
    device = get_device()
    ckpt_path = config.best_checkpoint_path

    if not ckpt_path.exists():
        found = list(config.checkpoint_dir.glob("**/*.pt")) + list(config.checkpoint_dir.glob("*.pt"))
        if found:
            ckpt_path = found[0]

    if not ckpt_path.exists():
        return None, None, config, device

    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    metadata_dim = checkpoint.get("metadata_dim", 47)
    model_name = checkpoint.get("model_name", config.backbone_name)

    model = FusionModel(
        backbone_name=model_name,
        metadata_dim=metadata_dim,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    processor = None
    if config.metadata_processor_path.exists():
        processor = MetadataProcessor.load(str(config.metadata_processor_path))

    return model, processor, config, device


@st.cache_resource
def get_eval_transform(image_size: int):
    return build_transforms(train=False, image_size=image_size)


def main():
    st.set_page_config(
        page_title="ISIC Medical AI | Skin Cancer Diagnostic Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Design Tokens & CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        .stApp { background-color: #0F172A; color: #F8FAFC; }
        
        .brand-header {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .brand-title { font-size: 2.2rem; font-weight: 700; color: #38BDF8; letter-spacing: -0.02em; }
        .brand-subtitle { font-size: 1.05rem; color: #94A3B8; margin-top: 0.5rem; }
        
        .card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .metric-value { font-size: 2.5rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.03em; }
        .metric-label { font-size: 0.875rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
        
        .badge-high {
            background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%);
            border: 1px solid #EF4444;
            color: #FEE2E2;
            padding: 1rem 1.25rem;
            border-radius: 10px;
            font-weight: 700;
            font-size: 1.05rem;
            text-align: center;
            margin-top: 1rem;
        }
        .badge-mod {
            background: linear-gradient(135deg, #78350F 0%, #92400E 100%);
            border: 1px solid #F59E0B;
            color: #FEF3C7;
            padding: 1rem 1.25rem;
            border-radius: 10px;
            font-weight: 700;
            font-size: 1.05rem;
            text-align: center;
            margin-top: 1rem;
        }
        .badge-low {
            background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
            border: 1px solid #10B981;
            color: #D1FAE5;
            padding: 1rem 1.25rem;
            border-radius: 10px;
            font-weight: 700;
            font-size: 1.05rem;
            text-align: center;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">🔬 ISIC Medical AI — Skin Cancer Diagnostic Assistant</div>
        <div class="brand-subtitle">Multimodal Deep Learning System integrating EfficientNetV2 visual feature maps with patient clinical metadata.</div>
    </div>
    """, unsafe_allow_html=True)

    model, processor, config, device = load_cached_artifacts()

    # History session state
    if "history" not in st.session_state:
        st.session_state.history = []

    st.sidebar.markdown("### 📋 Clinical Patient Metadata")
    with st.sidebar.form("clinical_metadata_form"):
        age = st.slider("Patient Age", min_value=1, max_value=100, value=45)
        sex = st.selectbox("Sex", options=["male", "female", "Unknown"])
        anatom_site = st.selectbox(
            "Anatomical Location",
            options=["torso", "lower extremity", "upper extremity", "head/neck", "palms/soles", "oral/genital", "Unknown"],
        )
        form_submitted = st.form_submit_button("⚡ Apply Patient Context", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Engine Diagnostics")
    st.sidebar.markdown(f"**Backbone Architecture**: `{config.backbone_name}`")
    st.sidebar.markdown(f"**Input Resolution**: `{config.image_size}x{config.image_size}`")
    st.sidebar.markdown(f"**Compute Hardware**: `{device}`")

    tabs = st.tabs(["🔍 Single Lesion Diagnosis", "📊 Prediction History", "⚡ System Diagnostics"])

    with tabs[0]:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📷 Upload Lesion Image")
            uploaded_file = st.file_uploader("Choose a dermoscopic image (JPEG, PNG, WEBP)...", type=["jpg", "jpeg", "png", "webp"])

            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption=f"Uploaded Image ({image.size[0]}x{image.size[1]} px)", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("⚡ Diagnostic Prediction")

            if uploaded_file and model is not None:
                with st.spinner("Analyzing visual feature maps & patient context..."):
                    transform = get_eval_transform(config.image_size)
                    augmented = transform(image=np.array(image))
                    image_tensor = augmented["image"].unsqueeze(0).to(device)

                    meta_df = pd.DataFrame([{
                        "isic_id": "UI_SAMPLE",
                        "age_approx": age,
                        "sex": sex,
                        "anatom_site_general": anatom_site,
                    }])

                    target_dim = getattr(model.metadata_mlp.net[0], "in_features", 47) if model else 47

                    if processor is not None and getattr(processor, "is_fitted", False):
                        meta_vec = processor.transform(meta_df)
                        if meta_vec.shape[1] > target_dim:
                            meta_vec = meta_vec[:, :target_dim]
                        elif meta_vec.shape[1] < target_dim:
                            pad_width = target_dim - meta_vec.shape[1]
                            meta_vec = np.pad(meta_vec, ((0, 0), (0, pad_width)), mode="constant")
                    else:
                        meta_vec = np.zeros((1, target_dim), dtype=np.float32)

                    meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        logits = model(image_tensor, meta_tensor)
                        prob = float(torch.sigmoid(logits).item())

                    st.markdown(f'<div class="metric-value">{prob * 100:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Malignancy Probability</div>', unsafe_allow_html=True)
                    st.progress(prob)

                    if prob >= 0.70:
                        risk_tag = "HIGH RISK"
                        st.markdown('<div class="badge-high">⚠️ HIGH RISK MALIGNANCY — Urgent Biopsy Indicated</div>', unsafe_allow_html=True)
                    elif prob >= 0.35:
                        risk_tag = "MODERATE RISK"
                        st.markdown('<div class="badge-mod">⚡ MODERATE RISK — Sequential Dermoscopy Recommended</div>', unsafe_allow_html=True)
                    else:
                        risk_tag = "LOW RISK"
                        st.markdown('<div class="badge-low">✅ LOW RISK — Typical Benign Characteristics</div>', unsafe_allow_html=True)

                    # Save to prediction history
                    st.session_state.history.append({
                        "filename": uploaded_file.name,
                        "probability": f"{prob * 100:.1f}%",
                        "risk_level": risk_tag,
                        "age": age,
                        "sex": sex,
                        "site": anatom_site,
                    })

                    # Side-by-side Grad-CAM Comparison
                    st.markdown("---")
                    show_cam = st.toggle("Show Grad-CAM Heatmap Comparison", value=True)
                    if show_cam:
                        gradcam = GradCAM(model)
                        heatmap = gradcam.generate(image_tensor, meta_tensor)
                        overlay = overlay_heatmap_on_image(image, heatmap)
                        
                        cam_col1, cam_col2 = st.columns(2)
                        with cam_col1:
                            st.image(image, caption="Original Dermoscopic Scan", use_container_width=True)
                        with cam_col2:
                            st.image(overlay, caption="Grad-CAM Activation Map", use_container_width=True)

            elif model is None:
                st.warning("Model checkpoint not found. Please verify trained model existence.")
            else:
                st.info("Upload a dermoscopic lesion image to run diagnosis.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("📋 Session Prediction History")
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("No predictions recorded in this session yet.")

    with tabs[2]:
        st.subheader("⚙️ Hardware & Environment Health")
        st.json({
            "PyTorch Version": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "Device": str(device),
            "Config Backbone": config.backbone_name,
            "Target Image Size": config.image_size,
        })


if __name__ == "__main__":
    main()

