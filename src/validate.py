from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.metrics import compute_pauc


def validate(model, dataloader, criterion=None, device=None, use_metadata: bool = True):
    """Validate the model and compute pAUC and ROC-AUC metrics.

    Supports both legacy ``(image, label)`` and Phase 3 ``(image, metadata, label)``
    dataloaders.  When *use_metadata* is ``True`` the model receives both the
    image tensor and the metadata tensor (Phase 3 fusion model).
    """
    model.eval()
    device = device or next(model.parameters()).device
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0
    probabilities = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Support both (image, meta, label) and legacy (image, label)
            if len(batch) == 3:
                images, metadata, labels = batch
            else:
                images, labels = batch
                metadata = None

            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            if use_metadata and metadata is not None:
                metadata = metadata.to(device)
                logits = model(images, metadata)
            else:
                logits = model(images)

            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
            targets.append(labels.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(targets) if targets else np.array([])
    y_score = np.concatenate(probabilities) if probabilities else np.array([])
    valid_mask = y_true >= 0
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    # Compute pAUC (partial AUC with max_fpr=0.1, the ISIC competition metric)
    if y_true.size > 0 and np.unique(y_true).size > 1:
        pauc = compute_pauc(y_true, y_score, max_fpr=0.1)
        roc_auc = roc_auc_score(y_true, y_score)
    else:
        pauc = float("nan")
        roc_auc = float("nan")

    average_loss = total_loss / max(total_samples, 1)
    return {"loss": average_loss, "roc_auc": roc_auc, "pauc": pauc}
