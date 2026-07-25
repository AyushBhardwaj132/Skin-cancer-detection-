from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    device: str = Field(..., json_schema_extra={"example": "cuda"})
    backbone: str = Field(..., json_schema_extra={"example": "tf_efficientnetv2_m"})
    image_size: int = Field(..., json_schema_extra={"example": 384})


class PredictionResponse(BaseModel):
    isic_id: str = Field(..., json_schema_extra={"example": "ISIC_0015657"})
    malignancy_probability: float = Field(..., json_schema_extra={"example": 0.8742})
    risk_level: str = Field(..., json_schema_extra={"example": "HIGH RISK"})
    recommendation: str = Field(..., json_schema_extra={"example": "Urgent dermatological biopsy recommended."})
    confidence_score: float = Field(..., json_schema_extra={"example": 0.92})


class ErrorResponse(BaseModel):
    detail: str = Field(..., json_schema_extra={"example": "Invalid file uploaded or missing metadata."})
