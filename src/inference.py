from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import ISICDataset
from src.ensemble import blend_predictions
from src.fusion_model import FusionModel
from src.metadata import MetadataProcessor
from src.model import build_model
from src.patient_features import enrich_metadata
from src.transforms import build_transforms, build_tta_transforms
from src.utils import ensure_dir, get_device, load_checkpoint


def _load_inference_frame(config: Config) -> pd.DataFrame:
    for candidate in (config.test_metadata_path, config.sample_submission_path):
        if candidate.exists():
            frame = pd.read_csv(candidate)
            if config.image_id_column in frame.columns:
                return frame.copy()
    raise FileNotFoundError(
        f"Could not find a test metadata or sample submission file under {config.data_dir}"
    )


def predict_single_model(
    config: Config,
    checkpoint_path: str | Path,
    use_tta: bool = False,
    use_fp16: bool = True,
) -> np.ndarray:
    """Predict probabilities for test data using a single model checkpoint (with optional TTA & FP16)."""
    device = get_device()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    use_metadata = checkpoint.get("use_metadata", config.use_metadata)
    metadata_dim = checkpoint.get("metadata_dim", 1)

    inference_frame = _load_inference_frame(config)
    meta_features = None

    if use_metadata:
        if config.use_patient_features and "patient_id" in inference_frame.columns:
            inference_frame = enrich_metadata(inference_frame, patient_col="patient_id")

        processor_path = config.metadata_processor_path
        if processor_path.exists():
            processor = MetadataProcessor.load(str(processor_path))
            meta_features = processor.transform(inference_frame)

    # --- Setup TTA transforms or single transform ---
    if use_tta:
        transforms_list = build_tta_transforms(image_size=config.image_size)
    else:
        transforms_list = [build_transforms(train=False, image_size=config.image_size)]

    model_name = checkpoint.get("model_name", config.backbone_name)

    if use_metadata and metadata_dim > 1:
        model = FusionModel(
            backbone_name=model_name,
            metadata_dim=metadata_dim,
            pretrained=False,
            metadata_hidden=config.metadata_mlp_hidden,
            metadata_output=config.metadata_mlp_output,
        ).to(device)
    else:
        model = build_model(model_name=model_name, pretrained=False, num_classes=1).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Apply torch.compile if supported and on CUDA
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    tta_predictions = []

    for tfm in transforms_list:
        dataset = ISICDataset(
            inference_frame,
            config.test_image_dir,
            transform=tfm,
            target_column=None,
            image_id_column=config.image_id_column,
            metadata_features=meta_features,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        pass_preds = []
        with torch.no_grad():
            for batch in dataloader:
                images, metadata, _ = batch
                images = images.to(device)

                with torch.cuda.amp.autocast(enabled=(use_fp16 and device.type == "cuda")):
                    if use_metadata and metadata_dim > 1:
                        metadata = metadata.to(device)
                        logits = model(images, metadata)
                    else:
                        logits = model(images)

                probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
                pass_preds.extend(probs.tolist())

        tta_predictions.append(np.array(pass_preds, dtype=np.float32))

    # Average predictions across TTA passes
    final_preds = np.mean(tta_predictions, axis=0)
    return final_preds


def predict(
    config: Config | None = None,
    checkpoint_path: str | Path | None = None,
    use_tta: bool = True,
    method: str = "rank",
) -> pd.DataFrame:
    """Run inference across model checkpoints (single or ensemble) with TTA & FP16."""
    config = config or Config()
    ensure_dir(config.prediction_dir)

    inference_frame = _load_inference_frame(config)
    image_ids = inference_frame[config.image_id_column].tolist()

    # Search for all available checkpoints across directories
    checkpoints_to_run = []
    if checkpoint_path is not None:
        checkpoints_to_run.append(Path(checkpoint_path))
    else:
        # Look for checkpoints in backbone folders or root checkpoint dir
        for backbone in config.ensemble_backbones:
            bb_dir = config.get_backbone_checkpoint_dir(backbone)
            found = list(bb_dir.glob("*.pt"))
            checkpoints_to_run.extend(found)

        # Fallback to root checkpoint directory
        if not checkpoints_to_run:
            found = list(config.checkpoint_dir.glob("*.pt"))
            checkpoints_to_run.extend(found)

    if not checkpoints_to_run:
        raise FileNotFoundError(f"No model checkpoints (.pt) found under {config.checkpoint_dir}")

    print(f"Found {len(checkpoints_to_run)} checkpoint(s) for inference:")
    for ckpt in checkpoints_to_run:
        print(f"  - {ckpt}")

    predictions_dict = {}
    for idx, ckpt in enumerate(checkpoints_to_run):
        print(f"Running inference for model {idx+1}/{len(checkpoints_to_run)} ({ckpt.name})...")
        preds = predict_single_model(
            config,
            checkpoint_path=ckpt,
            use_tta=use_tta,
            use_fp16=config.use_fp16,
        )
        predictions_dict[ckpt.stem] = preds

    # Blend predictions
    if len(predictions_dict) == 1:
        final_probs = list(predictions_dict.values())[0]
    else:
        print(f"Ensembling {len(predictions_dict)} model predictions using '{method}' averaging...")
        final_probs = blend_predictions(predictions_dict, method=method)

    submission = pd.DataFrame(
        {
            config.image_id_column: image_ids,
            config.target_column: final_probs,
        }
    )

    submission.to_csv(config.submission_path, index=False)
    submission.to_csv(config.ensemble_prediction_path, index=False)
    print(f"Submission successfully saved to {config.submission_path} ({len(submission)} rows)")
    return submission


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    image_size: int = 384,
    metadata_dim: int = 50,
) -> None:
    """Export PyTorch FusionModel checkpoint to ONNX format for fast inference engines."""
    try:
        import onnx
    except ImportError:
        print("onnx library not installed. Install with 'pip install onnx' for ONNX export.")
        return

    device = torch.device("cpu")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "tf_efficientnetv2_m")

    model = FusionModel(
        backbone_name=model_name,
        metadata_dim=metadata_dim,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_img = torch.randn(1, 3, image_size, image_size, device=device)
    dummy_meta = torch.randn(1, metadata_dim, device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_img, dummy_meta),
        str(output_path),
        input_names=["images", "metadata"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "metadata": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    predict()
