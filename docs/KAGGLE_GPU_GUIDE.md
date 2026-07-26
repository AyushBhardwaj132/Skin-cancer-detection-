# ⚡ Kaggle GPU Setup & Training Guide

This guide details how to run full competition training for the **ISIC 2024 Skin Cancer Detection System** on Kaggle GPUs (NVIDIA P100 / T4 / V100).

---

## 📌 Prerequisites & Hardware Requirements

- **Kaggle Account**: Signed in with GPU quota enabled.
- **Accelerator**: **GPU P100** or **GPU T4 x2**.
- **Dataset**: Attach official competition dataset `isic-2024-challenge`.

---

## 🚀 Step-by-Step Kaggle Execution Guide

### Step 1: Upload / Push Project to GitHub
Ensure all latest code updates are committed and pushed:
```bash
git add .
git commit -m "feat: Add Kaggle GPU training compatibility"
git push origin main
```

---

### Step 2: Create New Kaggle Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook**.
3. In the right panel under **Notebook Options**:
   - **Accelerator**: Select **GPU P100** or **GPU T4 x2**.
   - **Persistence**: Select **Variables & Files**.
4. In the right panel under **Input Data**, click **+ Add Data**:
   - Search for `isic-2024-challenge` (or official competition dataset).
   - Click **Add**.

---

### Step 3: Run Training Notebook Cells

Run the following commands directly inside notebook cells:

#### Cell 1: Clone Repository & Install Dependencies
```bash
!git clone https://github.com/AyushBhardwaj132/Skin-cancer-detection-.git /kaggle/working/Skin-cancer-detection-
%cd /kaggle/working/Skin-cancer-detection-
!pip install -q -r requirements.txt
```

#### Cell 2: Verify GPU Hardware & CUDA Acceleration
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    !nvidia-smi
```

#### Cell 3: Execute Full 5-Fold GroupKFold Competition Training
```bash
!python train_kaggle.py --all-folds --epochs 10
```

#### Cell 4: Run Inference & Generate Submission CSV
```bash
!python main.py infer --method rank
```

#### Cell 5: Compress Checkpoints for Local Download
```bash
!zip -q -r /kaggle/working/isic2024_checkpoints.zip /kaggle/working/outputs/checkpoints/
```

---

## ⏱️ Expected Training Runtime

| Environment / Hardware | Batch Size | Time / Epoch | Time / Fold (10 Epochs) | Total 5-Fold Ensemble |
|---|:---:|:---:|:---:|:---:|
| **NVIDIA GPU P100 (16GB)** | 32 | ~1.2 min | ~12 min | **~60 minutes** |
| **NVIDIA GPU T4 (16GB)** | 32 | ~1.6 min | ~16 min | **~80 minutes** |
| **Laptop CPU (Intel i7)** | 8 | ~6.1 min | ~61 min | ~5.0 hours |

---

## 📥 Downloading Checkpoints & Local Deployment

1. After training completes, locate `isic2024_checkpoints.zip` in the `/kaggle/working/` output panel.
2. Download `isic2024_checkpoints.zip` to your local machine.
3. Extract checkpoints into your local workspace directory:
   ```bash
   unzip isic2024_checkpoints.zip -d outputs/
   ```
4. Launch local Streamlit clinical UI or FastAPI backend using downloaded GPU checkpoints:
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## 🔒 Verification & Zero Logic Alteration Guarantee

The Kaggle GPU configuration uses the **exact same model architecture** (`FusionModel`), loss function (`FocalLoss`), patient feature engineering (`Ugly Duckling` scores), and metric evaluations (`pAUC@0.1`) as local training. Only execution hardware flags (`AMP FP16`, `cudnn.benchmark`) are toggled dynamically.
