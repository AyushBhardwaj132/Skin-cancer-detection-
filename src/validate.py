from __future__ import annotations

import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from src.metrics import compute_pauc, find_optimal_threshold


def _time_stage(stage_name: str, fn, *args, **kwargs):
    """Executes a metric function while measuring elapsed time and warning if > 30s."""
    print(f"  [VALIDATION] Starting {stage_name}...", flush=True)
    sys.stdout.flush()
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"  [VALIDATION] Finished {stage_name} in {elapsed:.3f}s", flush=True)
    if elapsed > 30.0:
        print(f"  [WARN] Stage '{stage_name}' took {elapsed:.1f}s (>30s limit)!", flush=True)
    sys.stdout.flush()
    return res, elapsed


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module | None = None,
    device: torch.device | str | None = None,
    use_metadata: bool = True,
    use_tta: bool = False,
    log_interval: int = 100,
) -> dict[str, float]:
    """Validate model with step-by-step trace logs, timing, and metric profiling."""
    t_val_start = time.perf_counter()
    print("  [TRACE 1/12] Entered validate() function in src/validate.py", flush=True)
    sys.stdout.flush()

    print("  [TRACE 2/12] Setting model.eval()", flush=True)
    model.eval()
    sys.stdout.flush()

    device = device or next(model.parameters()).device
    print(f"  [TRACE 3/12] Target device: {device}", flush=True)
    sys.stdout.flush()

    if torch.cuda.is_available():
        print("  [TRACE 4/12] Synchronizing CUDA prior to validation loop...", flush=True)
        sys.stdout.flush()
        torch.cuda.synchronize()
        print("  [TRACE 4/12] CUDA synchronized successfully.", flush=True)
        sys.stdout.flush()

    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0
    probabilities = []
    targets = []

    print("  [TRACE 5/12] Getting DataLoader length...", flush=True)
    sys.stdout.flush()
    total_batches = len(dataloader)
    print(f"  [TRACE 5/12] DataLoader has {total_batches} total batches", flush=True)
    sys.stdout.flush()

    print("  [TRACE 6/12] Creating DataLoader iterator...", flush=True)
    sys.stdout.flush()
    t_iter_start = time.perf_counter()
    try:
        val_iter = iter(dataloader)
        t_iter_end = time.perf_counter()
        print(f"  [TRACE 6/12] DataLoader iterator created in {t_iter_end - t_iter_start:.3f}s", flush=True)
        sys.stdout.flush()
    except Exception as iter_err:
        print(f"  [ERROR] Failed to create DataLoader iterator: {iter_err}", flush=True)
        sys.stdout.flush()
        raise iter_err

    print("  [TRACE 7/12] Entering torch.no_grad() validation loop...", flush=True)
    sys.stdout.flush()

    t_loop_start = time.perf_counter()
    batch_idx = 0

    with torch.no_grad():
        while True:
            batch_idx += 1
            print(f"  [TRACE BATCH {batch_idx}/{total_batches}] Requesting next batch from DataLoader worker...", flush=True)
            sys.stdout.flush()

            t_fetch0 = time.perf_counter()
            try:
                batch = next(val_iter)
                t_fetch1 = time.perf_counter()
                print(f"  [TRACE BATCH {batch_idx}/{total_batches}] Batch fetched in {t_fetch1 - t_fetch0:.4f}s", flush=True)
                sys.stdout.flush()
            except StopIteration:
                print(f"  [TRACE BATCH] Reached end of DataLoader at batch {batch_idx - 1}", flush=True)
                sys.stdout.flush()
                break
            except Exception as fetch_err:
                print(f"  [ERROR BATCH {batch_idx}] Exception while fetching batch: {fetch_err}", flush=True)
                sys.stdout.flush()
                raise fetch_err

            print(f"  [TRACE BATCH {batch_idx}] Transferring batch to device ({device})...", flush=True)
            sys.stdout.flush()
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

            print(f"  [TRACE BATCH {batch_idx}] Executing model forward pass...", flush=True)
            sys.stdout.flush()
            t_fwd0 = time.perf_counter()
            if use_tta:
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
            else:
                logits = model(images, metadata) if (use_metadata and metadata is not None) else model(images)
                batch_probs = torch.sigmoid(logits)

            t_fwd1 = time.perf_counter()
            print(f"  [TRACE BATCH {batch_idx}] Forward pass completed in {t_fwd1 - t_fwd0:.4f}s", flush=True)
            sys.stdout.flush()

            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            print(f"  [TRACE BATCH {batch_idx}] Converting tensors to CPU numpy arrays...", flush=True)
            sys.stdout.flush()
            probabilities.append(batch_probs.detach().cpu().numpy().reshape(-1))
            targets.append(labels.detach().cpu().numpy().reshape(-1))

            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                elapsed_loop = time.perf_counter() - t_loop_start
                print(f"  [VALIDATION] Batch {batch_idx}/{total_batches} processed ({elapsed_loop:.1f}s elapsed)", flush=True)
                sys.stdout.flush()

    t_loop_end = time.perf_counter()
    loop_elapsed = t_loop_end - t_loop_start
    print(f"  [TRACE 8/12] Validation loop finished in {loop_elapsed:.2f}s ({total_samples} samples)", flush=True)
    sys.stdout.flush()

    print("  [TRACE 9/12] Concatenating targets and probabilities...", flush=True)
    sys.stdout.flush()
    y_true = np.concatenate(targets) if targets else np.array([])
    y_score = np.concatenate(probabilities) if probabilities else np.array([])
    valid_mask = y_true >= 0
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    pauc = float("nan")
    roc_auc = float("nan")
    opt_thresh = 0.5
    opt_f1 = 0.0

    if y_true.size > 0 and np.unique(y_true).size > 1:
        # 1. pAUC Computation
        print("  [TRACE 10/12] Before pAUC computation...", flush=True)
        sys.stdout.flush()
        pauc, t_pauc = _time_stage("pAUC computation", compute_pauc, y_true, y_score, max_fpr=0.1)
        print(f"  [TRACE 10/12] After pAUC computation: pAUC = {pauc:.4f} ({t_pauc:.3f}s)", flush=True)
        sys.stdout.flush()

        # 2. ROC-AUC Computation
        print("  [TRACE 11/12] Before ROC-AUC computation...", flush=True)
        sys.stdout.flush()
        roc_auc, t_roc = _time_stage("ROC-AUC computation", roc_auc_score, y_true, y_score)
        roc_auc = float(roc_auc)
        print(f"  [TRACE 11/12] After ROC-AUC computation: ROC-AUC = {roc_auc:.4f} ({t_roc:.3f}s)", flush=True)
        sys.stdout.flush()

        # 3. Optimal Threshold Search
        print("  [TRACE 12/12] Before optimal threshold search...", flush=True)
        sys.stdout.flush()
        (opt_thresh, opt_f1), t_thresh = _time_stage("Optimal threshold search", find_optimal_threshold, y_true, y_score, metric="f1")
        print(f"  [TRACE 12/12] After optimal threshold search: opt_thresh = {opt_thresh:.2f}, opt_f1 = {opt_f1:.4f} ({t_thresh:.3f}s)", flush=True)
        sys.stdout.flush()
    else:
        print("  [WARN] Skipping metric computations (only 1 class present in validation ground truth).", flush=True)
        sys.stdout.flush()

    total_val_elapsed = time.perf_counter() - t_val_start
    print(f"  [VALIDATION] Total validation stage completed in {total_val_elapsed:.2f}s", flush=True)
    sys.stdout.flush()

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
