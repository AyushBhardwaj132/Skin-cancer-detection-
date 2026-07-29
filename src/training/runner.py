from __future__ import annotations

import os
import sys
import gc
import time
from pathlib import Path
import torch

from src.config.config import Config
from src.config.validation import ensure_valid_config
from src.training.state import TrainingState
from src.training.hf_backup import HuggingFaceBackup
from src.data.caching import MetadataCacheManager
from src.inference.oof import generate_oof_predictions
from src.inference.predictor import predict


def configure_hardware_optimizations(config: Config) -> None:
    """Configures PyTorch hardware acceleration, matmul precision, and CuDNN benchmarks."""
    if torch.cuda.is_available():
        # Set TensorCore matrix multiplication precision for modern GPUs (T4 / P100 / V100 / A100)
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

        torch.backends.cudnn.benchmark = getattr(config, "cudnn_benchmark", True)
        if config.num_workers == 0 and sys.platform != "win32":
            config.num_workers = 4
    else:
        config.use_fp16 = False
        config.num_workers = 0


def run_full_competition_pipeline(
    config: Config | None = None,
    requested_fold: int | None = None,
    resume: bool = True,
) -> None:
    """Master Competition Training Pipeline.
    
    Executes automatic 5-fold GroupKFold training, auto-resumes completed folds/epochs,
    caches metadata, uploads best models to Hugging Face, generates OOF predictions,
    and creates final competition submission.
    """
    config = config or Config()
    ensure_valid_config(config)
    configure_hardware_optimizations(config)

    hf_backup = HuggingFaceBackup(repo_id=config.hf_repo_id) if getattr(config, "hf_enabled", True) else None
    if hf_backup and hf_backup.is_available:
        hf_backup.perform_self_test(test_upload=True)

    # Step 1: Pre-cache metadata & patient features to eliminate startup latency
    if config.use_metadata and config.train_metadata_path.exists():
        print("\n[FAST STARTUP] Checking metadata cache...")
        _ = MetadataCacheManager.load_or_compute_enriched_metadata(
            config.train_metadata_path, config, verbose=True
        )

    # Step 2: Load training state for auto-resume
    training_state = TrainingState.load(config.output_dir)

    # Determine fold loop list
    if requested_fold is not None and requested_fold >= 0:
        folds_to_run = [requested_fold]
        print(f"\n[EXECUTION MODE] Single-Fold Override Active -> Running Fold {requested_fold}")
    else:
        folds_to_run = list(range(config.n_splits))
        print(f"\n[EXECUTION MODE] Automatic 5-Fold Competition Cross-Validation Active -> Folds {folds_to_run}")

    print("=" * 80)
    print(f"ISIC 2024 COMPETITION PIPELINE — STARTING TRAINING LOOP ({len(folds_to_run)} folds)")
    print(f"  Backbone:           {config.backbone_name}")
    print(f"  Image Size:         {config.image_size}x{config.image_size}")
    print(f"  Batch Size:         {config.batch_size}")
    print(f"  Total Epochs:       {config.num_epochs}")
    print(f"  Auto-Resume:        {resume}")
    print(f"  Completed Folds:    {training_state.completed_folds}")
    print("=" * 80 + "\n")

    # Import train function dynamically to prevent circular imports
    from src.train import train

    for fold in folds_to_run:
        # Auto-Resume Check: skip fold if already completed
        if resume and fold in training_state.completed_folds:
            print(f"\n{'='*80}")
            print(f"  [AUTO-RESUME] Fold {fold}/{config.n_splits - 1} ALREADY COMPLETED. Skipping.")
            print(f"{'='*80}\n")
            continue

        print(f"\n{'='*80}")
        print(f"  STARTING FOLD {fold}/{config.n_splits - 1} | Backbone: {config.backbone_name}")
        print(f"{'='*80}\n")

        # Execute training for fold
        try:
            res = train(config, fold_idx=fold, resume=resume, hf_backup=hf_backup)
            best_pauc = res.get("best_val_pauc", 0.0)
            best_ckpt = res.get("best_checkpoint", "")

            # Mark fold completed in state
            training_state.mark_fold_completed(fold, best_pauc=best_pauc, best_ckpt_path=best_ckpt)
            training_state.save(config.output_dir)

            # Asynchronous Hugging Face Backup
            if hf_backup and best_ckpt and Path(best_ckpt).exists():
                print(f"  [HF BACKUP] Triggering non-blocking background upload for Fold {fold} best model...")
                hf_backup.upload_checkpoint_async(
                    local_path=best_ckpt,
                    fold_idx=fold,
                    model_name=config.backbone_name,
                )

        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Training manually stopped by user on Fold {fold}. State saved cleanly.")
            training_state.save(config.output_dir)
            sys.exit(0)
        except Exception as err:
            print(f"\n[ERROR] Exception encountered on Fold {fold}: {err}")
            training_state.save(config.output_dir)
            raise err
        finally:
            # Memory safety: cleanup GPU VRAM and run garbage collection between folds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("ALL FOLDS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    # Step 3: Automatic Out-Of-Fold (OOF) Prediction & Competition CV Metric Evaluation
    try:
        generate_oof_predictions(config)
    except Exception as oof_err:
        print(f"[WARN] Automatic OOF prediction generation failed: {oof_err}")

    # Step 4: Automatic Test Set Inference & Submission Generation
    test_exists = (config.test_metadata_path.exists() or config.sample_submission_path.exists())
    if test_exists:
        print("\n[AUTOMATIC INFERENCE] Test data detected. Generating ensemble submission.csv...")
        try:
            predict(config, use_tta=config.use_tta, method=config.ensemble_method)
            print("[AUTOMATIC INFERENCE] submission.csv created successfully [OK]!")
        except Exception as inf_err:
            print(f"[WARN] Automatic test inference failed: {inf_err}")
    else:
        print("\n[INFO] No test dataset found. Skipping automatic submission.csv generation.")

    print("\n" + "=" * 80)
    print("ISIC 2024 PIPELINE RUN FINISHED CLEANLY")
    print("=" * 80 + "\n")
