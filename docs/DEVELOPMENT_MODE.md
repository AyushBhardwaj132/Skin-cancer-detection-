# ISIC 2024 — Development Mode

## Purpose

Development Mode provides a **real, trained AI model** that can be demonstrated on a CPU-only laptop. It uses the **exact same architecture** (FusionModel with image backbone + metadata MLP) as the production competition pipeline — only the computation is reduced:

- Smaller dataset (1,000 training + 300 validation images)
- Fewer epochs (5)
- Lower image resolution (224x224 vs 384x384)
- Smaller backbone (EfficientNetV2-S vs EfficientNetV2-M)
- Smaller batch size (8)

Everything else — metadata processing, patient features, loss function, optimizer, checkpoint format, inference pipeline, Grad-CAM, Streamlit UI, FastAPI — remains **architecturally identical**.

---

## How It Works

### Architecture (identical to production)

```
Input Image (224x224)
       |
  EfficientNetV2-S Backbone (timm)
       |
  Image Features (1280-dim)
       |                           Patient Metadata
       |                                |
       |                         MetadataProcessor
       |                          (StandardScaler + OneHotEncoder)
       |                                |
       |                         MetadataMLP
       |                          (input -> 256 -> 128)
       |                                |
       +---------- Concatenate --------+
                      |
               Fusion Classifier
            (1408 -> 512 -> 128 -> 1)
                      |
                 Sigmoid -> Probability
```

### Training Pipeline

1. **Dataset**: Balanced subset from the full ISIC 2024 dataset (no image duplication)
2. **Metadata**: Full patient feature engineering + ugly duckling scores
3. **Loss**: Focal Loss (alpha=0.75, gamma=2.0) for class imbalance
4. **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
5. **Scheduler**: CosineAnnealingLR
6. **Validation**: Every epoch with ROC-AUC, pAUC, precision, recall, F1

---

## How to Train

### Step 1: Create Development Splits (one-time)

```bash
python scripts/create_dev_splits.py
```

This creates:
- `data/dev_train.csv` (500 benign + 500 malignant)
- `data/dev_validation.csv` (150 benign + 150 malignant)

### Step 2: Train

```bash
python train_dev.py
```

Optional overrides:
```bash
python train_dev.py --epochs 10 --batch-size 4 --lr 0.0001
python train_dev.py --backbone tf_efficientnetv2_s --image-size 224
```

### Step 3: Launch Streamlit

```bash
streamlit run app/streamlit_app.py
```

The app will automatically detect the development checkpoint and default to it.

---

## How to Switch Models

In the Streamlit sidebar, use the **Model Selection** dropdown:

- **Development Model** — trained on balanced subset, suitable for demo
- **Production Model** — from competition training (if available)

The selector displays:
- Checkpoint filename
- Backbone architecture
- Training epoch
- Training date
- Validation ROC-AUC
- Validation pAUC

---

## Configuration

All development settings are in `configs/dev_config.yaml`:

| Parameter | Value | Production Value |
|---|---|---|
| Backbone | EfficientNetV2-S | EfficientNetV2-M |
| Image Size | 224 | 384 |
| Batch Size | 8 | 32 |
| Epochs | 5 | 10 |
| Learning Rate | 1e-4 | 1e-4 |
| Loss | Focal | Focal |
| Metadata Fusion | Yes | Yes |
| Patient Features | Yes | Yes |
| EMA | No (CPU) | Yes |
| Mixed Precision | No (CPU) | Yes |
| Workers | 0 | 4 |

---

## Expected CPU Training Time

On a modern laptop CPU (Intel i7/i9, AMD Ryzen 7/9):

| Phase | Time |
|---|---|
| Data loading + metadata | ~30 seconds |
| Per epoch (1000 images) | ~3-8 minutes |
| 5 epochs total | ~15-40 minutes |
| Evaluation + plots | ~1 minute |

Total: **~20-45 minutes** depending on CPU speed.

---

## Expected Accuracy

With only 1,000 training images (500 per class) at 224px:

| Metric | Expected Range |
|---|---|
| ROC-AUC | 0.70 - 0.90 |
| pAUC@0.1 | 0.05 - 0.30 |
| Accuracy | 0.65 - 0.85 |
| F1 Score | 0.60 - 0.80 |

These are demonstration-quality metrics. Production models trained on the full dataset with GPU would achieve significantly higher scores.

---

## Checkpoints

Checkpoints are saved to `outputs/checkpoints/dev/`:

```
outputs/checkpoints/dev/
    best_model.pt          # Best validation ROC-AUC
    epoch_1.pt             # Per-epoch checkpoints
    epoch_2.pt
    epoch_3.pt
    epoch_4.pt
    epoch_5.pt
    training_history.csv   # Training metrics log
    dev_metadata_processor.joblib  # Fitted metadata processor
```

Production checkpoints in `outputs/checkpoints/` are **never modified**.

---

## Evaluation Outputs

After training, evaluation plots are saved to `outputs/evaluation/dev/`:

- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `probability_histogram.png`
- `calibration_curve.png`
- `evaluation_metrics.json`

---

## Limitations

1. **Small training set** — only 1,000 images vs 37,724 in the full dataset
2. **Lower resolution** — 224x224 vs 384x384
3. **Smaller backbone** — EfficientNetV2-S vs EfficientNetV2-M/L
4. **No EMA** — disabled for CPU speed
5. **No MixUp/CutMix** — disabled for CPU speed
6. **No mixed precision** — CPU does not benefit from FP16
7. **Potential overfitting** — small dataset + powerful model
8. **Metadata may overfit** — limited patient diversity in small subset

---

## Future GPU Training

When GPU is available, upgrade to production training:

```bash
# Full competition training on GPU
python train.py --fold 0 --epochs 10
python train.py --all-folds

# Or full ensemble
python main.py train-ensemble
```

GPU training uses:
- Full 37,724-image dataset
- 384x384 resolution
- EfficientNetV2-M/L backbones
- Mixed precision (FP16)
- EMA
- MixUp + CutMix augmentation
- 5-fold GroupKFold cross-validation

---

## Files

| File | Purpose |
|---|---|
| `configs/dev_config.yaml` | Development configuration |
| `scripts/create_dev_splits.py` | Generate balanced dev CSVs |
| `train_dev.py` | Development training entry point |
| `data/dev_train.csv` | Training subset metadata |
| `data/dev_validation.csv` | Validation subset metadata |
| `docs/dataset_report.md` | Dataset verification report |
| `docs/DEVELOPMENT_MODE.md` | This document |
