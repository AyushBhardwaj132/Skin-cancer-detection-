# ISIC 2024 Technical Methodology & Validation Report

## Abstract
Skin cancer classification poses unique challenges due to severe class imbalance, extreme visual similarity between benign dysplastic nevi and early-stage melanoma, and inter-patient variance. This report details the technical methodology, feature engineering strategies, model architectures, and validation results of our multimodal solution for the ISIC 2024 Challenge.

---

## 1. Problem Formulations & Competition Metric

### 1.1 Objective
Binary classification of 3D skin lesion images predicting whether a given lesion is malignant ($y=1$) or benign ($y=0$).

### 1.2 Evaluation Metric: Partial AUC (pAUC)
Standard ROC-AUC evaluates the entire False Positive Rate (FPR) spectrum from $0.0$ to $1.0$. However, in clinical screening, high false-positive rates lead to unnecessary invasive biopsies. The ISIC 2024 challenge evaluates models using **partial AUC (pAUC)** restricted to $\text{FPR} \in [0.0, 0.1]$:

$$\text{pAUC} = \frac{1}{\text{max\_fpr}} \int_{0}^{\text{max\_fpr}} \text{TPR}(f) \, df$$

---

## 2. Feature Engineering & Metadata Processing

### 2.1 Metadata Normalization
We extract 34 continuous numerical parameters and 4 categorical attributes. Continuous features are median-imputed and scaled using `StandardScaler`. Categorical features are one-hot encoded.

### 2.2 Patient-Level Aggregations
For every patient ($P_i$), we aggregate lesion statistics across all registered lesions:
- Lesion Count ($N_{lesions}$)
- Mean and Max Lesion Diameter
- Mean $L^*$ channel color brightness and standard deviation

### 2.3 Ugly Duckling Score
The "Ugly Duckling" sign is a clinical heuristic: a lesion that looks different from a patient's other lesions is more likely malignant. We compute the Euclidean distance of lesion $j$ from patient centroid $\bar{\mathbf{x}}_i$:

$$D_{ij} = \|\mathbf{x}_{ij} - \bar{\mathbf{x}}_i\|_2$$

---

## 3. Multimodal Architecture & Ensembling

### 3.1 Multimodal Fusion Model
The visual backbone processes images resized to $384 \times 384$. Simultaneously, raw metadata passes through `MetadataMLP` ($D \to 256 \to 128$). Features are concatenated ($2048 + 128 = 2176$) and passed to a multi-stage classification head.

### 3.2 Loss Function: Focal Loss
To address class imbalance (malignant samples $<1\%$), we replace BCE with Focal Loss ($\alpha=0.75, \gamma=2.0$):

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

---

## 4. Experimental Results

Our progressive phase-by-phase performance evolution:

1. **Baseline**: pAUC 0.1040
2. **GroupKFold + Focal Loss**: pAUC 0.1380
3. **Metadata Fusion & Patient Features**: pAUC 0.1850
4. **15-Model Ensemble (Rank Averaging)**: pAUC **0.2310**
5. **Distilled Student Model**: pAUC 0.2180 (Inference: 18ms)
