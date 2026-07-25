from __future__ import annotations

from contextlib import asynccontextmanager
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from src.config import Config
from src.models.fusion_model import FusionModel
from src.data.metadata import MetadataProcessor
from src.data.transforms import build_transforms
from src.utils import get_device, load_checkpoint, get_logger
from src.utils.xai import GradCAM, overlay_heatmap_on_image
from src.api.schemas import HealthResponse, PredictionResponse, ErrorResponse

logger = get_logger("ISIC_API")

config = Config()
device = get_device()
model: FusionModel | None = None
processor: MetadataProcessor | None = None
metadata_dim: int = 47


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, metadata_dim
    ckpt_path = config.best_checkpoint_path
    if not ckpt_path.exists():
        found = list(config.checkpoint_dir.glob("**/*.pt")) + list(config.checkpoint_dir.glob("*.pt"))
        if found:
            ckpt_path = found[0]

    if ckpt_path.exists():
        logger.info(f"Loading FastAPI model checkpoint: {ckpt_path}")
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

    if config.metadata_processor_path.exists():
        processor = MetadataProcessor.load(str(config.metadata_processor_path))
        logger.info("Loaded tabular metadata processor.")
    yield


app = FastAPI(
    title="ISIC Skin Cancer Detection API",
    description="Enterprise REST API for predicting skin lesion malignancy probability & generating Grad-CAM explainability heatmaps.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Step 6 & 7 Security: Explicit allowed origins (no wildcard "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error processing request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."},
    )


@app.get("/", tags=["Health"])
def root():
    return {
        "name": "ISIC 2024 Skin Cancer Detection API",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "version": "2.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "backbone": config.backbone_name,
        "image_size": config.image_size,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Inference"],
)
async def predict_lesion(
    file: UploadFile = File(...),
    age_approx: float = Form(45.0),
    sex: str = Form("male"),
    anatom_site_general: str = Form("torso"),
):
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model checkpoint is not loaded.")

    if file.content_type not in config.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types: {config.allowed_mime_types}",
        )

    try:
        contents = await file.read()
        if len(contents) > config.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum limit of {config.max_upload_size_mb} MB.",
            )

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid or unreadable image file: {e}")

    transform = build_transforms(train=False, image_size=config.image_size)
    augmented = transform(image=np.array(image))
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    meta_df = pd.DataFrame([{
        "isic_id": "API_SAMPLE",
        "age_approx": age_approx,
        "sex": sex,
        "anatom_site_general": anatom_site_general,
    }])

    if processor is not None and getattr(processor, "is_fitted", False):
        meta_vec = processor.transform(meta_df)
    else:
        meta_vec = np.zeros((1, metadata_dim), dtype=np.float32)

    meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(image_tensor, meta_tensor)
        prob = float(torch.sigmoid(logits).item())

    if prob >= 0.70:
        risk_level = "HIGH RISK"
        recommendation = "Urgent specialist dermatological biopsy recommended."
    elif prob >= 0.35:
        risk_level = "MODERATE RISK"
        recommendation = "Clinical examination and short-term sequential dermoscopy monitoring recommended."
    else:
        risk_level = "LOW RISK"
        recommendation = "Benign visual presentation. Standard annual skin check."

    confidence = float(np.abs(prob - 0.5) * 2.0)

    return {
        "isic_id": "API_SAMPLE",
        "malignancy_probability": round(prob, 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "confidence_score": round(confidence, 4),
    }


@app.post("/explain", tags=["Explainable AI (XAI)"])
async def explain_lesion(
    file: UploadFile = File(...),
    age_approx: float = Form(45.0),
    sex: str = Form("male"),
    anatom_site_general: str = Form("torso"),
):
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image file: {e}")

    transform = build_transforms(train=False, image_size=config.image_size)
    image_np = np.array(image)
    augmented = transform(image=image_np)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    meta_df = pd.DataFrame([{
        "isic_id": "API_SAMPLE",
        "age_approx": age_approx,
        "sex": sex,
        "anatom_site_general": anatom_site_general,
    }])
    if processor is not None and getattr(processor, "is_fitted", False):
        meta_vec = processor.transform(meta_df)
    else:
        meta_vec = np.zeros((1, metadata_dim), dtype=np.float32)

    meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).to(device)

    try:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate(image_tensor, meta_tensor)
        overlay_pil = overlay_heatmap_on_image(image, heatmap)

        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate Grad-CAM: {e}")
