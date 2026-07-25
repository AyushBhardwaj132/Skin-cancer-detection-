from __future__ import annotations

from contextlib import asynccontextmanager
import gc
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends, status
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
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

eval_transform = build_transforms(train=False, image_size=config.image_size)


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

    logger.info("Shutting down API service & cleaning resources...")
    model = None
    processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="ISIC Medical AI — Skin Cancer Detection REST API",
    description="Production-grade REST API service predicting skin lesion malignancy probabilities & generating Grad-CAM heatmaps.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "System diagnostics and API service readiness checks."},
        {"name": "Inference", "description": "Skin lesion image upload and clinical risk prediction endpoints."},
        {"name": "Explainable AI (XAI)", "description": "Grad-CAM visual feature map heatmap generation."},
    ],
    lifespan=lifespan,
)

# CORS Middleware
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


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Dependency Injection for Model Check
def get_loaded_model() -> FusionModel:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model checkpoint is not loaded. Service initializing or unavailable.",
        )
    return model


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error processing request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."},
    )


def _validate_image_bytes(contents: bytes) -> None:
    """Validates raw file header magic bytes to prevent spoofed content-types."""
    if len(contents) < 4:
        raise ValueError("File stream too short to be a valid image.")

    # Magic byte signatures: JPEG (\xFF\xD8\xFF), PNG (\x89PNG), WEBP (RIFF...WEBP)
    is_jpeg = contents.startswith(b"\xff\xd8\xff")
    is_png = contents.startswith(b"\x89PNG")
    is_webp = contents.startswith(b"RIFF") and b"WEBP" in contents[:16]

    if not (is_jpeg or is_png or is_webp):
        raise ValueError("Uploaded binary does not match JPEG, PNG, or WEBP magic-byte header signatures.")


def _run_prediction_sync(contents: bytes, age_approx: float, sex: str, anatom_site_general: str) -> float:
    _validate_image_bytes(contents)
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    augmented = eval_transform(image=np.array(image))
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    meta_df = pd.DataFrame([{
        "isic_id": "API_SAMPLE",
        "age_approx": age_approx,
        "sex": sex,
        "anatom_site_general": anatom_site_general,
    }])

    target_dim = getattr(model.metadata_mlp.net[0], "in_features", metadata_dim) if model else metadata_dim

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

    return prob


def _run_explain_sync(contents: bytes, age_approx: float, sex: str, anatom_site_general: str) -> bytes:
    _validate_image_bytes(contents)
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    augmented = eval_transform(image=np.array(image))
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    meta_df = pd.DataFrame([{
        "isic_id": "API_SAMPLE",
        "age_approx": age_approx,
        "sex": sex,
        "anatom_site_general": anatom_site_general,
    }])

    target_dim = getattr(model.metadata_mlp.net[0], "in_features", metadata_dim) if model else metadata_dim

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

    gradcam = GradCAM(model)
    heatmap = gradcam.generate(image_tensor, meta_tensor)
    overlay_pil = overlay_heatmap_on_image(image, heatmap)

    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


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
    active_model: FusionModel = Depends(get_loaded_model),
):
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

        prob = await run_in_threadpool(_run_prediction_sync, contents, age_approx, sex, anatom_site_general)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid or unreadable image file: {e}")

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
    active_model: FusionModel = Depends(get_loaded_model),
):
    try:
        contents = await file.read()
        png_bytes = await run_in_threadpool(_run_explain_sync, contents, age_approx, sex, anatom_site_general)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate Grad-CAM: {e}")


