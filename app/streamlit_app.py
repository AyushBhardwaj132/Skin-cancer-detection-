from __future__ import annotations

import io
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.config import Config
from src.fusion_model import FusionModel
from src.metadata import MetadataProcessor
from src.transforms import build_transforms
from src.utils import get_device, load_checkpoint
from src.xai import GradCAM, overlay_heatmap_on_image

# Streamlit Page Config
st.set_page_config(
    page_title="Skin Cancer Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 1rem; text-align: center; }
    .risk-high { background-color: #FEE2E2; border-left: 5px solid #EF4444; padding: 1rem; border-radius: 4px; }
    .risk-mod { background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 1rem; border-radius: 4px; }
    .risk-low { background-color: #D1FAE5; border-left: 5px solid #10B981; padding: 1rem; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_processor():
    config = Config()
    device = get_device()
    ckpt_path = config.best_checkpoint_path
    
    if not ckpt_path.exists():
        found = list(config.checkpoint_dir.glob("*.pt"))
        if found:
            ckpt_path = found[0]
            
    if not ckpt_path.exists():
        return None, None, config, device
        
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    metadata_dim = checkpoint.get("metadata_dim", 50)
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


def main():
    st.markdown('<div class="main-header">🔬 ISIC Skin Cancer Detection & Diagnostic Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multimodal AI fusing deep visual feature representations with patient clinical metadata.</div>', unsafe_allow_html=True)

    model, processor, config, device = load_model_and_processor()

    # Sidebar Clinical Metadata Inputs
    st.sidebar.header("📋 Patient Context Metadata")
    age = st.sidebar.slider("Patient Age (Years)", min_value=1, max_value=100, value=45)
    sex = st.sidebar.selectbox("Sex", options=["male", "female", "Unknown"])
    anatom_site = st.sidebar.selectbox(
        "Anatomical Location",
        options=["torso", "lower extremity", "upper extremity", "head/neck", "palms/soles", "oral/genital", "Unknown"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Configuration")
    st.sidebar.info(f"**Backbone**: {config.backbone_name}\n\n**Input Size**: {config.image_size}x{config.image_size}\n\n**Device**: {device}")

    # Main Uploader Panel
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Upload Skin Lesion Image")
        uploaded_file = st.file_uploader("Choose a JPG/PNG lesion image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Lesion Image", use_column_width=True)

    with col2:
        st.subheader("📊 Diagnostic Predictions & XAI")
        
        if uploaded_file is not None:
            if model is None:
                st.error("⚠️ Model checkpoint not found in `outputs/checkpoints/`. Please train a model first using `python main.py train`.")
                return

            with st.spinner("Analyzing lesion image & patient context..."):
                # Transform image
                image_np = np.array(image)
                transform = build_transforms(train=False, image_size=config.image_size)
                transformed = transform(image=image_np)
                img_tensor = transformed["image"].unsqueeze(0).to(device)

                # Metadata vector
                meta_df = pd.DataFrame([{"age_approx": age, "sex": sex, "anatom_site_general": anatom_site}])
                if processor is not None:
                    meta_vec = processor.transform(meta_df)
                else:
                    meta_vec = np.zeros((1, 50), dtype=np.float32)
                meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).to(device)

                # Model prediction
                with torch.no_grad():
                    logits = model(img_tensor, meta_tensor)
                    prob = float(torch.sigmoid(logits).item())

            # Display Gauge / Probability
            st.markdown(f"### Malignancy Probability: **{prob*100:.1f}%**")
            st.progress(prob)

            if prob >= 0.7:
                st.markdown(f'<div class="risk-high">🚨 <b>HIGH RISK OF MALIGNANCY</b><br>Strong visual and clinical features suggestive of melanoma. Professional dermatological evaluation strongly advised.</div>', unsafe_allow_html=True)
            elif prob >= 0.3:
                st.markdown(f'<div class="risk-mod">⚠️ <b>MODERATE RISK</b><br>Lesion exhibits atypical characteristics. Monitoring or follow-up recommended.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ <b>LOW RISK (LIKELY BENIGN)</b><br>Features align closely with benign nevi / seborrheic keratosis.</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Grad-CAM Toggle
            show_gradcam = st.checkbox("🔥 View Grad-CAM Explanation Heatmap", value=True)
            if show_gradcam:
                with st.spinner("Generating Grad-CAM attention heatmap..."):
                    try:
                        gradcam = GradCAM(model)
                        heatmap = gradcam.generate_heatmap(img_tensor, meta_tensor)
                        resized_img = cv2.resize(image_np, (config.image_size, config.image_size))
                        overlay = overlay_heatmap_on_image(resized_img, heatmap)
                        
                        col_cam1, col_cam2 = st.columns(2)
                        with col_cam1:
                            st.image(heatmap, caption="Grad-CAM Activation", use_column_width=True, clamp=True)
                        with col_cam2:
                            st.image(overlay, caption="Heatmap Overlay", use_column_width=True)
                    except Exception as e:
                        st.warning(f"Could not render Grad-CAM heatmap: {e}")

        else:
            st.info("👆 Please upload a skin lesion image on the left to view predictions.")


if __name__ == "__main__":
    main()
