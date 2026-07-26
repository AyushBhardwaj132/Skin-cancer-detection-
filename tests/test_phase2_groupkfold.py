import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.data.split import get_fold_dataframes
from src.training.losses import FocalLoss, get_loss_fn
from src.evaluation.metrics import compute_pauc


def test_groupkfold_patient_leakage(tmp_path):
    """Test that GroupKFold guarantees zero patient ID overlap between train and validation splits."""
    data = []
    for p in range(20):
        patient_id = f"P_{p:03d}"
        for l in range(np.random.randint(1, 5)):
            data.append({
                "isic_id": f"ISIC_{p}_{l}",
                "patient_id": patient_id,
                "target": np.random.choice([0, 1], p=[0.9, 0.1]),
            })
    df = pd.DataFrame(data)
    csv_path = tmp_path / "train-metadata.csv"
    df.to_csv(csv_path, index=False)

    for fold in range(5):
        train_df, val_df = get_fold_dataframes(csv_path, fold_idx=fold, n_splits=5)
        train_patients = set(train_df["patient_id"].unique())
        val_patients = set(val_df["patient_id"].unique())

        intersection = train_patients.intersection(val_patients)
        assert len(intersection) == 0, f"Patient leakage detected in fold {fold}: {intersection}"
        assert len(train_df) + len(val_df) == len(df), f"Row count mismatch in fold {fold}"


def test_focal_loss_forward_backward():
    """Test Focal Loss computation, alpha/gamma scaling, and gradient computation."""
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.randn(10, 1, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32).unsqueeze(1)

    loss = criterion(logits, targets)
    assert loss.item() > 0.0, "Focal loss should be positive"
    assert torch.isfinite(loss), "Focal loss must be finite"

    loss.backward()
    assert logits.grad is not None, "Gradients should be computed"
    assert not torch.isnan(logits.grad).any(), "Gradients should not contain NaNs"


def test_pauc_metric_calculation():
    """Test official Partial AUC (pAUC @ max_fpr=0.1) metric behavior."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.1, 0.05, 0.9, 0.85])

    pauc = compute_pauc(y_true, y_pred, max_fpr=0.1)
    assert 0.0 <= pauc <= 1.0, f"pAUC score out of bounds: {pauc}"

    # Single class edge case
    single_class_pauc = compute_pauc(np.zeros(10), y_pred, max_fpr=0.1)
    assert single_class_pauc == 0.0, "Single class should return 0.0 pAUC"
