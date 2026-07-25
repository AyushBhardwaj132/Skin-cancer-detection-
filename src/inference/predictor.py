from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import ISICDataset
from src.data.transforms import build_transforms, build_tta_transforms
from src.data.metadata import MetadataProcessor
from src.data.patient_features import enrich_metadata
from src.models.fusion_model import FusionModel
from src.models.model import build_model
from src.inference.ensemble import blend_predictions
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
    """Predict probabilities for test data using a single model checkpoint."""
    device = get_device()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    use_metadata = checkpoint.get("use_metadata", config.use_metadata)
    # Determine metadata dimension dynamically if metadata is used
    metadata_dim = checkpoint.get("metadata_dim", 1)
    inference_frame = _load_inference_frame(config)
    meta_features = None

    if use_metadata:
        if config.use_patient_features and "patient_id" in inference_frame.columns:
            inference_frame = enrich_metadata(inference_frame)

        processor_path = config.metadata_processor_path
        if processor_path.exists():
            processor = MetadataProcessor.load(str(processor_path))
            # Ensure processor is fitted; older checkpoints may lack the flag
            if not getattr(processor, "is_fitted", False):
                processor.fit(inference_frame)
            meta_features = processor.transform(inference_frame)
            # Align feature dimension with checkpoint expectation
            if meta_features.shape[1] > metadata_dim:
                meta_features = meta_features[:, :metadata_dim]
            elif meta_features.shape[1] < metadata_dim:
                # Pad with zeros if needed (unlikely)
                pad_width = metadata_dim - meta_features.shape[1]
                meta_features = np.pad(meta_features, ((0,0),(0,pad_width)), mode='constant')
        # Override metadata_dim with actual feature size if available
        if meta_features is not None:
            metadata_dim = meta_features.shape[1]

    tta_transforms = build_tta_transforms(image_size=config.image_size) if use_tta else [build_transforms(train=False, image_size=config.image_size)]

    model_name = checkpoint.get("model_name", config.backbone_name)
    if use_metadata and metadata_dim > 0:
        model = FusionModel(
            backbone_name=model_name,
            metadata_dim=metadata_dim,
            pretrained=False,
        ).to(device)
    else:
        model = build_model(model_name=model_name, pretrained=False, num_classes=1).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_pass_preds = []

    for tfa in tta_transforms:
        dataset = ISICDataset(
            df=inference_frame,
            image_dir=config.test_image_dir,
            transform=tfa,
            is_test=True,
            image_id_col=config.image_id_column,
            metadata_tensor=meta_features,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        pass_preds = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                if use_metadata and "metadata" in batch:
                    meta = batch["metadata"].to(device)
                    with torch.amp.autocast('cuda', enabled=(use_fp16 and device.type == "cuda")):
                        logits = model(images, meta)
                else:
                    with torch.amp.autocast('cuda', enabled=(use_fp16 and device.type == "cuda")):
                        logits = model(images)

                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                pass_preds.append(probs)

        all_pass_preds.append(np.concatenate(pass_preds, axis=0))

    return np.mean(all_pass_preds, axis=0)


def predict(
    config: Config,
    checkpoint_path: str | Path | None = None,
    use_tta: bool = True,
    method: str = "rank",
) -> pd.DataFrame:
    """Runs inference across all trained checkpoints (or single checkpoint) and returns submission DataFrame."""
    if checkpoint_path is not None:
        checkpoints = [Path(checkpoint_path)]
    else:
        checkpoints = list(config.checkpoint_dir.glob("**/*.pt")) + list(config.checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No trained checkpoints found under {config.checkpoint_dir}")

    print(f"Found {len(checkpoints)} checkpoint(s) for inference:")
    for ck in checkpoints:
        print(f"  - {ck}")

    predictions_list = []
    for idx, ck in enumerate(checkpoints, 1):
        print(f"Running inference for model {idx}/{len(checkpoints)} ({ck.name})...")
        preds = predict_single_model(config, checkpoint_path=ck, use_tta=use_tta, use_fp16=config.use_fp16)
        predictions_list.append(preds)

    blended_probs = blend_predictions(predictions_list, method=method)

    inference_frame = _load_inference_frame(config)
    submission_df = pd.DataFrame({
        config.image_id_column: inference_frame[config.image_id_column],
        config.target_column: blended_probs,
    })

    ensure_dir(config.submission_path.parent)
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission successfully saved to {config.submission_path} ({len(submission_df)} rows)")
    return submission_df
