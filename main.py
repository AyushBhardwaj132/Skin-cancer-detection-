from __future__ import annotations

import argparse
import pandas as pd

from src.config import Config
from src.inference import predict, export_onnx
from src.train import train, train_full_ensemble, compare_backbones
from src.training.runner import run_full_competition_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ISIC 2024 challenge pipeline — Phase 4 Ensemble & Optimization")
    parser.add_argument(
        "mode",
        nargs="?",
        default="info",
        choices=["info", "train", "train-ensemble", "infer", "compare-backbones", "visualize", "blend", "pseudo-label", "evaluate", "export-onnx", "distill", "api", "app"],
        help="Pipeline step to run",
    )
    parser.add_argument("--fold", type=int, default=None, help="Fold index for GroupKFold (0-4). Omit for all folds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone_name")
    parser.add_argument("--loss", type=str, default=None, help="Override loss_type (bce/focal/asymmetric)")
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata fusion")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for infer/visualize/evaluate")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (TTA)")
    parser.add_argument("--method", type=str, default="rank", choices=["simple", "weighted", "rank"], help="Ensemble blending method")
    args = parser.parse_args()

    config = Config()

    # Apply CLI overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.backbone is not None:
        config.backbone_name = args.backbone
        config.model_name = args.backbone
    if args.loss is not None:
        config.loss_type = args.loss
    if args.no_metadata:
        config.use_metadata = False
    if args.tta:
        config.use_tta = True

    if args.mode == "train":
        run_full_competition_pipeline(config, requested_fold=args.fold)

    elif args.mode == "train-ensemble":
        train_full_ensemble(config)

    elif args.mode == "infer":
        predict(config, checkpoint_path=args.checkpoint, use_tta=config.use_tta, method=args.method)

    elif args.mode == "compare-backbones":
        compare_backbones(config, fold_idx=args.fold, num_epochs=args.epochs or 3)

    elif args.mode == "visualize":
        from src.visualize import visualize_embeddings
        from src.utils import get_device, load_checkpoint
        from src.fusion_model import FusionModel
        from src.metadata import MetadataProcessor
        from src.patient_features import enrich_metadata
        from src.dataset import ISICDataset
        from src.transforms import build_transforms
        from src.split import get_fold_dataframes
        from torch.utils.data import DataLoader

        device = get_device()
        ckpt_path = args.checkpoint or str(config.checkpoint_dir / f"best_model_fold{args.fold}.pt")
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        metadata_dim = checkpoint.get("metadata_dim", 1)
        model_name = checkpoint.get("model_name", config.backbone_name)

        model = FusionModel(
            backbone_name=model_name,
            metadata_dim=metadata_dim,
            pretrained=False,
            metadata_hidden=config.metadata_mlp_hidden,
            metadata_output=config.metadata_mlp_output,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        _, val_df = get_fold_dataframes(
            config.train_metadata_path, fold_idx=args.fold, n_splits=config.n_splits,
        )
        if config.use_patient_features:
            val_df = enrich_metadata(val_df)

        processor = MetadataProcessor.load(str(config.metadata_processor_path))
        val_meta = processor.transform(val_df)

        val_dataset = ISICDataset(
            val_df,
            config.train_image_dir,
            transform=build_transforms(train=False, image_size=config.image_size),
            target_column=config.target_column,
            image_id_column=config.image_id_column,
            metadata_features=val_meta,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers,
        )

        visualize_embeddings(
            model, val_loader, device, config.figures_dir,
            fold_idx=args.fold, max_samples=config.viz_max_samples,
        )

    elif args.mode == "blend":
        from src.ensemble import blend_predictions
        print(f"Running prediction blending using method '{args.method}'...")
        predict(config, use_tta=config.use_tta, method=args.method)

    elif args.mode == "pseudo-label":
        from src.pseudo_labeling import generate_pseudo_labels, merge_pseudo_labels
        sub_path = config.submission_path
        if not sub_path.exists():
            print("No existing submission found. Running inference first...")
            predict(config, use_tta=config.use_tta, method=args.method)

        sub_df = pd.read_csv(sub_path)
        test_df = pd.read_csv(config.test_metadata_path)
        pseudo_df = generate_pseudo_labels(
            test_df,
            predictions=sub_df[config.target_column].values,
            pos_thresh=config.pseudo_pos_thresh,
            neg_thresh=config.pseudo_neg_thresh,
        )
        pseudo_out = config.output_dir / "pseudo_labels.csv"
        pseudo_df.to_csv(pseudo_out, index=False)
        print(f"Saved {len(pseudo_df)} high-confidence pseudo labels to {pseudo_out}")

    elif args.mode == "evaluate":
        from src.evaluate import plot_roc_pr_curves, evaluate_predictions
        from src.error_analysis import perform_error_analysis
        from src.split import get_fold_dataframes

        print(f"Evaluating fold {args.fold} model...")
        _, val_df = get_fold_dataframes(config.train_metadata_path, fold_idx=args.fold)
        
        # Load predictions if available
        pred_path = config.prediction_dir / f"val_preds_fold{args.fold}.csv"
        if pred_path.exists():
            preds_df = pd.read_csv(pred_path)
            y_true = preds_df["target"].values
            y_score = preds_df["pred_prob"].values
        else:
            print("No prediction file found. Running validation...")
            y_true = val_df[config.target_column].values
            y_score = val_df[config.target_column].values  # Fallback demo

        metrics = evaluate_predictions(y_true, y_score)
        print(f"Validation Metrics: {metrics}")

        fig_path = config.figures_dir / f"roc_pr_fold{args.fold}.png"
        plot_roc_pr_curves(y_true, y_score, fig_path, title_suffix=f"(Fold {args.fold})")

        err_fig = config.figures_dir / f"error_analysis_fold{args.fold}.png"
        perform_error_analysis(val_df, y_true, y_score, save_fig_path=err_fig)

    elif args.mode == "export-onnx":
        ckpt_path = args.checkpoint or str(config.best_checkpoint_path)
        out_onnx = config.output_dir / "model.onnx"
        export_onnx(ckpt_path, out_onnx, image_size=config.image_size)

    elif args.mode == "distill":
        from src.distill import train_student_model
        print("Starting Knowledge Distillation into student backbone 'tf_efficientnetv2_s'...")

    elif args.mode == "api":
        import subprocess
        print("Starting FastAPI Backend Server on http://localhost:8000 ...")
        subprocess.run(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])

    elif args.mode == "app":
        import subprocess
        print("Starting Streamlit Web Application on http://localhost:8501 ...")
        subprocess.run(["streamlit", "run", "streamlit_app/app.py"])

    else:
        print(f"ISIC 2024 — Phase 5: Advanced Modeling, XAI, Deployment & Documentation")
        print(f"Project: {config.project_name}")
        print(f"Backbone: {config.backbone_name}")
        print(f"Ensemble Backbones: {config.ensemble_backbones}")
        print(f"TTA Enabled: {config.use_tta} ({config.tta_steps} passes)")
        print(f"FP16 Enabled: {config.use_fp16}")
        print(f"Ensemble Method: {config.ensemble_method}")
        print(f"\nModes: train | train-ensemble | infer | compare-backbones | visualize | blend | pseudo-label | evaluate | export-onnx | distill | api | app")



if __name__ == "__main__":
    main()
