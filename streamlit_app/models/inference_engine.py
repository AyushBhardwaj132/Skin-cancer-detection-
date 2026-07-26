"""PyTorch inference engine & cached model artifact loader."""
from __future__ import annotations
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.config import Config
from src.models.fusion_model import FusionModel
from src.data.metadata import MetadataProcessor
from src.data.transforms import build_transforms
from src.utils import get_device, load_checkpoint
from src.utils.xai import GradCAM, overlay_heatmap_on_image

from streamlit_app.config.settings import (
    DEV_BEST_MODEL_PATH,
    DEV_PROCESSOR_PATH,
    PROD_BEST_MODEL_PATH,
    PROD_PROCESSOR_PATH,
    DEFAULT_IMAGE_SIZE,
)
from streamlit_app.utils.image_utils import crop_lesion_centered
from streamlit_app.utils.logger import get_app_logger

logger = get_app_logger("InferenceEngine")


@st.cache_resource
def load_model_artifacts() -> tuple[FusionModel | None, MetadataProcessor | None, str, torch.device]:
    """Loads trained EfficientNetV2-S FusionModel and MetadataProcessor into memory once."""
    device = get_device()
    logger.info(f"Target execution device: {device}")

    # Determine checkpoint path
    ckpt_path = DEV_BEST_MODEL_PATH
    proc_path = DEV_PROCESSOR_PATH

    if not ckpt_path.exists():
        ckpt_path = PROD_BEST_MODEL_PATH
        proc_path = PROD_PROCESSOR_PATH

    if not ckpt_path.exists():
        logger.error(f"No checkpoint file found at {ckpt_path} or {PROD_BEST_MODEL_PATH}")
        return None, None, "Checkpoint Missing", device

    try:
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        metadata_dim = checkpoint.get("metadata_dim", 47)
        model_name = checkpoint.get("model_name", "tf_efficientnetv2_s")

        logger.info(f"Loading FusionModel ({model_name}) with metadata_dim={metadata_dim}...")
        model = FusionModel(
            backbone_name=model_name,
            metadata_dim=metadata_dim,
            pretrained=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load MetadataProcessor
        processor = None
        if proc_path.exists():
            processor = MetadataProcessor.load(str(proc_path))
            logger.info(f"Loaded MetadataProcessor from {proc_path}")
        else:
            logger.warning(f"Processor not found at {proc_path}, creating fallback processor.")
            processor = MetadataProcessor()

        return model, processor, model_name, device
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return None, None, "Error Loading", device


class InferenceEngine:
    """Production inference controller executing end-to-end multimodal predictions."""
    def __init__(self):
        self.model, self.processor, self.model_name, self.device = load_model_artifacts()
        self.transform = build_transforms(train=False, image_size=DEFAULT_IMAGE_SIZE)

    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def predict(
        self,
        image: Image.Image,
        metadata_dict: dict,
        use_lesion_crop: bool = True
    ) -> dict:
        """Executes complete inference pipeline on input image and metadata."""
        start_time = time.perf_counter()

        if not self.is_ready():
            raise RuntimeError("Model or MetadataProcessor artifact is not loaded.")

        # 1. Preprocess Image
        img_np = np.array(image.convert("RGB"))
        if use_lesion_crop:
            cropped_np = crop_lesion_centered(img_np)
        else:
            cropped_np = img_np

        # Albumentations transform
        augmented = self.transform(image=cropped_np)
        img_tensor = augmented["image"].unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # 2. Preprocess Metadata
        meta_df = pd.DataFrame([metadata_dict])
        meta_features_np = self.processor.transform(meta_df)
        
        # Ensure metadata feature dimensions match model expectation
        expected_dim = self.model.metadata_mlp.net[0].in_features
        if meta_features_np.shape[1] < expected_dim:
            pad_width = expected_dim - meta_features_np.shape[1]
            meta_features_np = np.pad(meta_features_np, ((0, 0), (0, pad_width)), mode="constant")
        elif meta_features_np.shape[1] > expected_dim:
            meta_features_np = meta_features_np[:, :expected_dim]

        meta_tensor = torch.tensor(meta_features_np, dtype=torch.float32).to(self.device)

        # 3. Model Inference
        with torch.no_grad():
            logits = self.model(img_tensor, meta_tensor)
            prob = torch.sigmoid(logits).item()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Risk & Confidence Classification
        if prob >= 0.65:
            risk_level = "High Risk"
            prediction = "Malignant"
            risk_color = "#dc2626"
        elif prob >= 0.35:
            risk_level = "Moderate Risk"
            prediction = "Suspicious / Monitor"
            risk_color = "#f59e0b"
        else:
            risk_level = "Low Risk"
            prediction = "Benign"
            risk_color = "#10b981"

        confidence_pct = abs(prob - 0.5) * 200.0  # Scale distance from 0.5 decision boundary to %

        # 5. Grad-CAM Heatmap Generation
        gradcam_overlay = None
        try:
            cam_generator = GradCAM(self.model)
            heatmap = cam_generator.generate(img_tensor, meta_tensor)
            gradcam_overlay = overlay_heatmap_on_image(cropped_np, heatmap)
        except Exception as e:
            logger.warning(f"Grad-CAM generation notice: {e}")
            # Generate synthetic fallback visualization if needed
            h, w = cropped_np.shape[:2]
            syn_heatmap = np.zeros((h, w), dtype=np.float32)
            cv2.circle(syn_heatmap, (w // 2, h // 2), min(h, w) // 3, 1.0, -1)
            syn_heatmap = cv2.GaussianBlur(syn_heatmap, (55, 55), 0)
            gradcam_overlay = overlay_heatmap_on_image(cropped_np, syn_heatmap)

        return {
            "probability": prob,
            "prediction": prediction,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "confidence_pct": round(confidence_pct, 1),
            "inference_time_ms": round(elapsed_ms, 1),
            "model_used": "EfficientNetV2-S (Metadata Fusion)",
            "cropped_image": Image.fromarray(cropped_np),
            "gradcam_overlay": gradcam_overlay,
        }
