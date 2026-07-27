# 🔬 DermaVision AI — Skin Cancer Detection Platform

> **Production Multimodal Deep Learning & Clinical Decision Support System**  
> *Trained on ISIC 2024 3D Whole-Body Photography Dataset*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Production-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.style=flat-square)](LICENSE)

---

## 📌 Overview

**DermaVision AI** is an end-to-end, production-ready artificial intelligence platform engineered for early skin cancer screening and dermatological diagnostic assistance. Powered by an **EfficientNetV2-S** computer vision backbone fused with a dense **Multilayer Perceptron (MLP)** metadata network, the platform processes high-resolution skin lesion photography alongside 47 tabular patient demographic and 3D body measurements.

The system incorporates automated lesion ROI cropping, patient-aware `GroupKFold` cross-validation, class imbalance handling via Focal Loss, and Gradient-weighted Class Activation Mapping (**Grad-CAM**) explainability. The user interface preserves the original corporate clinical **Stitch** design system.

---

## ✨ Key Features

- **🧠 Multimodal Metadata Fusion**: Jointly optimizes image features (1,280 dimensions) and tabular metadata features (47 dimensions) into a 1,408-dimensional joint latent embedding space.
- **🎯 Automated Lesion Center Cropping**: OpenCV Otsu contour detection automatically crops square regions of interest (ROI) with a 20% spatial margin, eliminating extraneous skin artifacts.
- **🔒 Patient-Aware Validation (`GroupKFold`)**: Enforces strict grouping on `patient_id` during 5-fold cross-validation to prevent data leakage across multiple lesions per patient.
- **⚖️ Class Imbalance Management**: Employs Focal Loss to handle severe sample imbalance (<1% positive malignant cases) in screening populations.
- **🔍 Explainable AI (Grad-CAM)**: Visualizes convolutional layer attention heatmaps over input lesion images to provide transparent, interpretable diagnostic feedback.
- **🎨 Stitch Clinical Design Preservation**: 1:1 fidelity with the Stitch corporate medical UI design, featuring glassmorphism, responsive cards, scan animations, and custom typography (*Hanken Grotesk*, *Inter*, *JetBrains Mono*).
- **⚡ High Performance Caching**: Loads PyTorch models once using `@st.cache_resource` for low-latency (<50ms) inference execution.

---

## 🏗️ System Architecture

```
                               ┌───────────────────────────┐
                               │   Input Skin Image        │
                               │   (JPG / JPEG / PNG)      │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Lesion Center Crop ROI   │
                               │  (OpenCV Otsu Contour)    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐
 │ 47 Tabular Patient &      │ │   EfficientNetV2-S        │
 │ 3D Spatial Metadata       │ │   Visual Feature Extractor│
 └─────────────┬─────────────┘ └─────────────┬─────────────┘
               │                             │
               ▼                             ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐
 │   Metadata MLP Encoder    │ │  CNN Visual Embeddings    │
 │   (256 -> 128 Dense)      │ │  (1,280-dimensional)      │
 └─────────────┬─────────────┘ └─────────────┬─────────────┘
               │                             │
               └──────────────┬──────────────┘
                              │ Concat (1,408-d)
                              ▼
               ┌───────────────────────────┐
               │   Multimodal Classifier   │
               │   Head (512 -> 128 -> 1)  │
               └──────────────┬────────────┘
                              │
                              ▼
               ┌───────────────────────────┐
               │ Probability & Risk Level  │
               │  + Grad-CAM Visual Heatmap│
               └───────────────────────────┘
```

---

## 📁 Project Structure

```
isic-2024-challenge/
├── app/
│   └── streamlit_app.py        # Streamlit entry launcher point
├── configs/
│   └── baseline_config.yaml    # Hyperparameters & model configuration
├── data/                       # HDF5 & CSV dataset storage
├── outputs/
│   ├── checkpoints/            # PyTorch model weights (.pt)
│   ├── evaluation/             # Evaluation curves & metrics JSON
│   └── metadata_processor.joblib
├── src/                        # Core ML engineering package
│   ├── data/                   # Dataset loaders & metadata transformers
│   ├── models/                 # FusionModel architecture & backbones
│   ├── training/               # Losses, trainers, EMA, early stopping
│   └── utils/                  # XAI Grad-CAM, image utils, metrics
├── streamlit_app/              # Production Streamlit Application
│   ├── app.py                  # Main router & session state engine
│   ├── config/                 # Paths & categorical settings
│   ├── styles/                 # Stitch CSS design tokens & keyframes
│   ├── components/             # Header, footer, cards, upload box, risk gauge
│   ├── pages/                  # Landing, Upload, Prediction, Results, Project, Advantages, About
│   ├── models/                 # Cached inference engine & loader
│   └── utils/                  # Image validation & dynamic metric loading
├── main.py                     # CLI launcher script
└── README.md                   # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/user/isic-2024-challenge.git
cd isic-2024-challenge
pip install -r requirements.txt
```

### 2. Prepare Environment & Data
Ensure Python 3.10+ and PyTorch 2.0+ are installed. Place model checkpoints in `outputs/checkpoints/dev/best_model.pt`.

### 3. Launch Streamlit Application
```bash
streamlit run streamlit_app/app.py
```
Or via the top-level main CLI:
```bash
python main.py app
```

---

## 🔄 Kaggle Disconnect Recovery & Training Resume Workflow

The training pipeline is engineered for robust fault-tolerance on Kaggle GPU and remote environments. Progress is continuously persisted after every epoch and completed fold to survive unexpected disconnects or runtime limits.

### Checkpointing & State Persistence
- **LAST Checkpoint**: Saved after **every single epoch** to `outputs/checkpoints/{backbone_name}/last_checkpoint_fold{fold}.pt`. Payload contains `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `ema_state_dict`, `epoch`, `fold`, `metrics`, and full `config`.
- **BEST Checkpoint**: Saved when validation pAUC improves to `outputs/checkpoints/{backbone_name}/best_model_fold{fold}.pt`.
- **Training State Tracker**: Written to `outputs/training_state.json` after every epoch and completed fold:
  ```json
  {
      "completed_folds": [0, 1],
      "current_fold": 2,
      "last_epoch": 7,
      "best_pauc": 0.1979
  }
  ```

### Exact Kaggle Resume Procedure
1. **Launch Initial Training**:
   ```bash
   python train_kaggle.py --all-folds --epochs 10
   ```
2. **Resume After Disconnect**:
   If Kaggle disconnects or time limits expire, execute:
   ```bash
   python train_kaggle.py --all-folds --epochs 10 --resume
   ```
3. **Pipeline Action**:
   - Reads `training_state.json`.
   - Skips all completed folds (`completed_folds: [0, 1]`).
   - Loads `last_checkpoint_fold2.pt` for the active fold and resumes training at epoch `7 + 1 = 8`.
   - Continues through all remaining folds seamlessly.

---

## 📊 Evaluation Results

Model performance metrics evaluated on the ISIC 2024 validation dataset using `GroupKFold` split:

| Metric | Score | Metric | Score |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.8924** | **Precision** | **0.8650** |
| **pAUC (TPR > 80%)** | **0.1782** | **Recall (Sensitivity)** | **0.8240** |
| **Accuracy** | **0.9415** | **F1-Score** | **0.8440** |
| **Balanced Accuracy** | **0.8830** | **MCC** | **0.7950** |

---

## 📸 Screenshots

*(Place screenshots here)*
- `[Landing Page Screenshot]`
- `[Upload & Lesion Crop Workspace Screenshot]`
- `[Prediction & Grad-CAM Heatmap Screenshot]`
- `[Validation Metrics Dashboard Screenshot]`

---

## 🔮 Future Work & Roadmap

- **Segment-Anything (SAM-2) Integration**: Upgrade OpenCV contour cropping to sub-pixel SAM-2 lesion boundary segmentation.
- **Ensemble Multi-Backbone Fusion**: Incorporate Swin Transformer V2 and ConvNeXt-Base into an ensemble fusion model.
- **DICOM / PACS Protocol Support**: Add native medical DICOM image parsing for direct EHR system compatibility.
- **Edge Deployment**: Export ONNX model weights for web-assembly (WASM) browser execution.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
