from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from src.config import Config
from src.fusion_model import FusionModel
from src.metadata import MetadataProcessor
from src.transforms import build_transforms
from src.utils import get_device, load_checkpoint
from src.xai import GradCAM, overlay_heatmap_on_image

app = FastAPI(
    title="ISIC Skin Cancer Detection API",
    description="REST API for predicting skin lesion malignancy probability & generating Grad-CAM explainability heatmaps.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
config = Config()
device = get_device()
model = None
processor = None
metadata_dim = 1


@app.on_event("startup")
def load_artifacts():
    global model, processor, metadata_dim
    ckpt_path = config.best_checkpoint_path
    if not ckpt_path.exists():
        # Fallback search
        found = list(config.checkpoint_dir.glob("*.pt"))
        if found:
            ckpt_path = found[0]
            
    if ckpt_path.exists():
        print(f"Loading API model from {ckpt_path}...")
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
        
    if config.metadata_processor_path.exists():
        processor = MetadataProcessor.load(str(config.metadata_processor_path))
        print("Loaded metadata processor for API.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "backbone": config.backbone_name,
        "image_size": config.image_size,
    }


@app.post("/predict")
async def predict_lesion(
    file: UploadFile = File(...),
    age_approx: float = Form(45.0),
    sex: str = Form("male"),
    anatom_site_general: str = Form("torso"),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a model first.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Process image
    transform = build_transforms(train=False, image_size=config.image_size)
    image_np = np.array(image)
    transformed = transform(image=image_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    # Process metadata
    meta_df = pd.DataFrame([{
        "age_approx": age_approx,
        "sex": sex,
        "anatom_site_general": anatom_site_general,
    }])
    
    if processor is not None:
        meta_vec = processor.transform(meta_df)
    else:
        meta_vec = np.zeros((1, metadata_dim), dtype=np.float32)

    meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(img_tensor, meta_tensor)
        prob = float(torch.sigmoid(logits).item())

    # Determine risk level
    if prob >= 0.7:
        risk_level = "High Risk (Suspicious)"
    elif prob >= 0.3:
        risk_level = "Moderate Risk (Monitor)"
    else:
        risk_level = "Low Risk (Likely Benign)"

    return {
        "filename": file.filename,
        "malignancy_probability": round(prob, 4),
        "risk_level": risk_level,
        "metadata_received": {
            "age": age_approx,
            "sex": sex,
            "anatom_site": anatom_site_general,
        }
    }


@app.post("/explain")
async def explain_lesion(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    image_np = np.array(image)
    transform = build_transforms(train=False, image_size=config.image_size)
    transformed = transform(image=image_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    # Generate Grad-CAM heatmap
    try:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate_heatmap(img_tensor)
        overlay = overlay_heatmap_on_image(cv2.resize(image_np, (config.image_size, config.image_size)), heatmap)
        
        # Encode overlay as PNG
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {e}")
