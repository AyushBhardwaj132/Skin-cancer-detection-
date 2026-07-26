"""App configuration and directory path resolutions."""
from __future__ import annotations
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STREAMLIT_APP_DIR = BASE_DIR / "streamlit_app"

# Model Checkpoints & Processors
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints" / "dev"
DEV_BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
DEV_PROCESSOR_PATH = CHECKPOINT_DIR / "dev_metadata_processor.joblib"

PROD_BEST_MODEL_PATH = BASE_DIR / "outputs" / "checkpoints" / "best_model.pt"
PROD_PROCESSOR_PATH = BASE_DIR / "outputs" / "metadata_processor.joblib"

# Evaluation Outputs
EVALUATION_METRICS_PATH = BASE_DIR / "outputs" / "evaluation" / "exp3_preprocessing" / "evaluation_metrics.json"
DEV_EVALUATION_METRICS_PATH = BASE_DIR / "outputs" / "evaluation" / "dev" / "evaluation_metrics.json"
EVALUATION_DIR = BASE_DIR / "outputs" / "evaluation" / "exp3_preprocessing"

# Architecture Configuration
MODEL_ARCHITECTURE = "EfficientNetV2-S"
BACKBONE_NAME = "tf_efficientnetv2_s"
DEFAULT_IMAGE_SIZE = 384
METADATA_DIM = 47

# Metadata Features List & Categorical Controls
CATEGORICAL_FIELDS = {
    "sex": ["male", "female", "unknown"],
    "anatom_site_general": [
        "head/neck",
        "upper extremity",
        "lower extremity",
        "torso",
        "palms/soles",
        "anterior torso",
        "posterior torso",
        "unknown"
    ],
    "image_type": ["overview", "close-up", "dermoscopy"],
    "tbp_tile_type": ["3D: full body", "3D: torso", "3D: head", "3D: arm", "3D: leg"],
    "tbp_lv_location": ["Torso Front", "Torso Back", "Arm Left", "Arm Right", "Leg Left", "Leg Right", "Head"],
    "tbp_lv_location_simple": ["Torso", "Arm", "Leg", "Head"]
}

DEFAULT_METADATA_VALUES = {
    "age_approx": 55.0,
    "sex": "male",
    "anatom_site_general": "lower extremity",
    "image_type": "overview",
    "tbp_tile_type": "3D: full body",
    "tbp_lv_location": "Leg Right",
    "tbp_lv_location_simple": "Leg",
    "clin_size_long_diam_mm": 4.5,
    "tbp_lv_areaMM2": 15.2,
    "tbp_lv_perimeterMM": 14.8,
    "tbp_lv_area_perim_ratio": 1.02,
    "tbp_lv_color_std_mean": 12.4,
    "tbp_lv_eccentricity": 0.45,
    "tbp_lv_nevi_confidence": 0.88,
    "tbp_lv_norm_border": 2.1,
    "tbp_lv_norm_color": 3.4,
    "tbp_lv_symm_2axis": 0.15,
    "tbp_lv_symm_2axis_angle": 45.0,
    "tbp_lv_A": 15.2,
    "tbp_lv_Aext": 14.8,
    "tbp_lv_B": 10.1,
    "tbp_lv_Bext": 9.8,
    "tbp_lv_C": 5.4,
    "tbp_lv_Cext": 5.2,
    "tbp_lv_H": 22.1,
    "tbp_lv_Hext": 21.8,
    "tbp_lv_L": 62.4,
    "tbp_lv_Lext": 61.9,
    "tbp_lv_deltaA": 2.1,
    "tbp_lv_deltaB": 1.8,
    "tbp_lv_deltaL": 3.2,
    "tbp_lv_deltaLB": 4.1,
    "tbp_lv_deltaLBnorm": 1.2,
    "tbp_lv_minorAxisMM": 3.2,
    "tbp_lv_radial_color_std_max": 2.8,
    "tbp_lv_stdL": 1.9,
    "tbp_lv_stdLExt": 2.0,
    "tbp_lv_x": -120.5,
    "tbp_lv_y": 450.2,
    "tbp_lv_z": 30.1,
}
