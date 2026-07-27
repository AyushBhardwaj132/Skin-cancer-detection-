from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from src.metrics import compute_pauc, find_optimal_threshold


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module | None = None,
    device: torch.device | str | None = None,
    use_metadata: bool = True,
    use_tta: bool = False,
) -> dict[str, float]:
    """Validate model with optional GPU-native TTA and optimal threshold search."""
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
            if isinstance(batch, dict):
                images = batch["image"].to(device, non_blocking=True)
                metadata = batch["metadata"].to(device, non_blocking=True) if ("metadata" in batch and batch["metadata"] is not None) else None
                labels = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)
            elif isinstance(batch, (tuple, list)):
                if len(batch) == 3:
                    images, metadata, labels = batch
                else:
                    images, labels = batch
                    metadata = None
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
                if metadata is not None:
                    metadata = metadata.to(device, non_blocking=True)
            else:
                raise TypeError(f"Unsupported batch type in validate: {type(batch)}")

            if use_tta:
                # GPU-native 4-view TTA: Original, HFlip, VFlip, Both Flips
                aug_images = [
                    images,
                    torch.flip(images, dims=[3]),
                    torch.flip(images, dims=[2]),
                    torch.flip(images, dims=[2, 3]),
                ]
                probs_list = []
                for img_aug in aug_images:
                    logits_aug = model(img_aug, metadata) if (use_metadata and metadata is not None) else model(img_aug)
                    probs_list.append(torch.sigmoid(logits_aug))

                batch_probs = torch.stack(probs_list, dim=0).mean(dim=0)
                logits = model(images, metadata) if (use_metadata and metadata is not None) else model(images)
            else:
                logits = model(images, metadata) if (use_metadata and metadata is not None) else model(images)
                batch_probs = torch.sigmoid(logits)

            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            probabilities.append(batch_probs.detach().cpu().numpy().reshape(-1))
            targets.append(labels.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(targets) if targets else np.array([])
    y_score = np.concatenate(probabilities) if probabilities else np.array([])
    valid_mask = y_true >= 0
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if y_true.size > 0 and np.unique(y_true).size > 1:
        pauc = compute_pauc(y_true, y_score, max_fpr=0.1)
        roc_auc = float(roc_auc_score(y_true, y_score))
        opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_score, metric="f1")
    else:
        pauc = float("nan")
        roc_auc = float("nan")
        opt_thresh = 0.5
        opt_f1 = 0.0

    average_loss = total_loss / max(total_samples, 1)
    return {
        "loss": average_loss,
        "roc_auc": roc_auc,
        "pauc": pauc,
        "optimal_threshold": opt_thresh,
        "f1_optimal": opt_f1,
        "y_true": y_true,
        "y_score": y_score,
    }
