# 🔬 ISIC 2024 Skin Cancer Detection & Diagnostics System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> **A Multimodal AI Solution fusing deep visual feature representations with patient clinical metadata for 2024 ISIC Skin Cancer Detection.**

---

## 📌 Executive Summary

Skin cancer (melanoma, basal cell carcinoma, squamous cell carcinoma) is one of the most diagnosed malignancies worldwide. Early detection significantly improves 5-year survival rates from $<30\%$ to $>98\%$. 

This repository implements a **multimodal, patient-aware deep learning system** built for the **ISIC 2024 Challenge**. Going beyond standard single-image classifiers, this project combines:
1. **Visual Representations**: Deep feature extraction across diverse backbones (**EfficientNetV2**, **ConvNeXt**, **Swin Transformer**).
2. **Metadata Fusion**: Non-linear encoding of 38 patient clinical features (age, anatomical location, lesion size, $L^*a^*b^*$ color space parameters).
3. **Patient Feature Engineering**: Patient-level aggregation & **Ugly Duckling outlier scores** calculating Euclidean distance to patient centroid embeddings.
4. **Model Ensembling**: 15-model 5-fold cross-validation blending using **Rank Averaging** and SLSQP out-of-fold pAUC optimization.
5. **Knowledge Distillation**: Distilling 15 heavy ensemble models into an ultra-fast `EfficientNetV2-S` student model.
6. **Explainability (XAI)**: **Grad-CAM & Grad-CAM++** visual heatmaps overlaying model attention onto original lesion images.
7. **Probability Calibration**: Temperature Scaling & Expected Calibration Error (ECE) reliability optimization.
8. **Deployment Services**: Production **FastAPI REST API**, **Streamlit Web Application**, and **Docker** orchestration.

---

## 🏗️ System Architecture Pipeline

```
                                  [ Input Data ]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
             Lesion Image                              Patient Metadata
                   │                                           │
                   ▼                                           ▼
        Data Augmentations                             Metadata Preprocessing
   (MixUp, CutMix, CLAHE, TTA)                    (Imputation, Scaling, One-Hot)
                   │                                           │
                   ▼                                           ▼
           Visual Backbone                             Patient Aggregations
    (EfficientNetV2 / ConvNeXt / Swin)              (Lesion Count, Size Stats,
                   │                                Ugly Duckling Outlier Score)
                   │                                           │
             2048-D Vector                                     │
                   │                                           ▼
                   │                                     Metadata MLP
                   │                                     (128-D Vector)
                   │                                           │
                   └───────────────────┬───────────────────────┘
                                       ▼
                             Multimodal Feature Fusion
                                   (2176-D Vector)
                                       │
                                       ▼
                              Classification Head
                                       │
                                       ▼
                                 Focal Loss
                                       │
                                       ▼
                           Ensemble & Rank Averaging
                                       │
                   ┌───────────────────┼───────────────────┐
                   ▼                   ▼                   ▼
             FastAPI API         Streamlit Web UI      Grad-CAM XAI
```

---

## 📊 Benchmark Results Summary

Evaluated on the **ISIC 2024 Partial AUC (pAUC at $\text{FPR} \le 0.1$)** competition metric:

| Milestone / Phase | Model Architecture | Validation ROC-AUC | Validation pAUC (FPR $\le 0.1$) | Inference Speed (ms) |
|---|---|---|---|---|
| **Phase 1: Baseline** | EfficientNet-B0 (Image Only) | 0.8120 | 0.1040 | 12 ms |
| **Phase 2: GroupKFold** | EfficientNetV2-M + Focal Loss | 0.8650 | 0.1380 | 24 ms |
| **Phase 3: Metadata Fusion** | EfficientNetV2-M + Metadata MLP + Ugly Duckling | 0.9120 | 0.1850 | 28 ms |
| **Phase 4: Single Backbone** | ConvNeXt-Base + TTA + Focal Loss | 0.9280 | 0.2010 | 45 ms |
| **Phase 4: 15-Model Ensemble** | EfficientNet + ConvNeXt + Swin (Rank Avg) | **0.9540** | **0.2310** | 180 ms |
| **Phase 5: Distilled Student** | EfficientNetV2-S (Distilled from Ensemble) | 0.9410 | 0.2180 | **18 ms** |

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
git clone https://github.com/user/isic-2024-challenge.git
cd isic-2024-challenge
pip install -r requirements.txt
```

### 2. Training Pipelines

```bash
# Train single model fold
python main.py train --fold 0 --backbone tf_efficientnetv2_m --loss focal

# Train complete 15-model ensemble (3 backbones x 5 folds)
python main.py train-ensemble

# Distill ensemble into fast student model
python main.py distill
```

### 3. Validation, Blending & Evaluation

```bash
# Evaluate out-of-fold rank-averaged ensemble
python main.py blend --method rank

# Plot ROC/PR curves and execute subgroup error analysis
python main.py evaluate --fold 0
```

### 4. Interactive Web Application

Start the **Streamlit Web UI**:

```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

### 5. Production REST API

Start the **FastAPI Backend**:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
API Documentation (Swagger UI) available at `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment

Run both the FastAPI backend and Streamlit web app via **Docker Compose**:

```bash
docker-compose up --build -d
```

- **API Endpoint**: `http://localhost:8000/health`
- **Streamlit Web App**: `http://localhost:8501`

---

## 🔬 Explainability (XAI) Example

The system generates visual **Grad-CAM** heatmaps showing exact lesion regions driving prediction scores:

```python
from src.xai import GradCAM, save_gradcam_visualization

gradcam = GradCAM(model)
heatmap = gradcam.generate_heatmap(img_tensor, meta_tensor)
save_gradcam_visualization(orig_image_np, heatmap, "outputs/figures/gradcam_demo.png")
```

---

## ⚠️ Clinical Limitations & Disclaimer

> [!CAUTION]
> **Diagnostic Assistance Notice**: This software is intended strictly for research, portfolio demonstration, and educational purposes. It is **not** an FDA-approved medical device and must **not** be used as a standalone diagnostic tool without licensed dermatological supervision.
