from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


def _detect_data_dir() -> Path:
    """Robustly auto-detects local vs Kaggle competition dataset input directory."""
    if os.getenv("DATA_DIR") and Path(os.getenv("DATA_DIR")).exists():
        return Path(os.getenv("DATA_DIR"))

    kaggle_paths = [
        Path("/kaggle/input/competitions/isic-2024-challenge"),
        Path("/kaggle/input/isic-2024-challenge"),
        Path("/kaggle/input/isic-2024"),
    ]
    for kp in kaggle_paths:
        if kp.exists() and (kp / "train-metadata.csv").exists():
            return kp

    # Dynamic search under /kaggle/input/ for any directory containing train-metadata.csv
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for sub_dir in kaggle_root.glob("**/*"):
            if sub_dir.is_dir() and (sub_dir / "train-metadata.csv").exists():
                return sub_dir

    return Path("data")


def _detect_output_dir() -> Path:
    if os.getenv("OUTPUT_DIR"):
        return Path(os.getenv("OUTPUT_DIR"))
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/outputs")
    return Path("outputs")


@dataclass(slots=True)
class Config:
    """Production configuration dataclass with environment-aware default paths."""
    project_name: str = "ISIC2024"
    data_dir: Path = field(default_factory=_detect_data_dir)
    output_dir: Path = field(default_factory=_detect_output_dir)
    
    # Metadata & Image Directories
    train_metadata_name: str = "train-metadata.csv"
    test_metadata_name: str = "test-metadata.csv"
    sample_submission_name: str = "sample_submission.csv"
    train_image_dir_name: str = "train-image"
    test_image_dir_name: str = "test-image"
    best_checkpoint_name: str = "best_model.pt"
    submission_name: str = "submission.csv"

    # Model Hyperparameters
    model_name: str = "tf_efficientnetv2_m"
    backbone_name: str = "tf_efficientnetv2_m"
    image_size: int = 384
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_size: float = 0.2
    seed: int = 42
    target_column: str = "target"
    image_id_column: str = "isic_id"

    # Phase 2: GroupKFold & Early Stopping
    n_splits: int = 5
    early_stopping_patience: int = 5
    pauc_max_fpr: float = 0.1

    # Phase 3: Patient & Metadata Fusion
    use_metadata: bool = True
    use_patient_features: bool = True
    use_ugly_duckling: bool = True
    metadata_mlp_hidden: int = 256
    metadata_mlp_output: int = 128

    # Loss Functions & Augmentations
    loss_type: str = "focal"  # focal, bce, weighted_bce
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    use_advanced_augs: bool = True
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0

    # Phase 4: Optimization & Ensembling
    ensemble_backbones: tuple[str, ...] = ("tf_efficientnetv2_l", "convnext_base", "swin_base_patch4_window12_384")
    use_tta: bool = True
    tta_steps: int = 4
    ensemble_method: str = "rank"  # simple, weighted, rank
    pseudo_pos_thresh: float = 0.99
    pseudo_neg_thresh: float = 0.01
    multi_gpu_mode: str = "auto"  # auto, single, dataparallel, ddp
    checkpoint_batch_interval: int = 500
    use_fp16: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Visualization & Security
    viz_max_samples: int = 5000
    max_upload_size_mb: int = 15
    allowed_mime_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")

    @property
    def train_metadata_path(self) -> Path:
        return self.data_dir / self.train_metadata_name

    @property
    def test_metadata_path(self) -> Path:
        return self.data_dir / self.test_metadata_name

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / self.sample_submission_name

    @property
    def train_image_dir(self) -> Path:
        return self.data_dir / self.train_image_dir_name

    @property
    def test_image_dir(self) -> Path:
        return self.data_dir / self.test_image_dir_name

    @property
    def train_image_hdf5_path(self) -> Path:
        return self.data_dir / "train-image.hdf5"

    @property
    def test_image_hdf5_path(self) -> Path:
        return self.data_dir / "test-image.hdf5"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def prediction_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.best_checkpoint_name

    @property
    def submission_path(self) -> Path:
        return self.prediction_dir / self.submission_name

    @property
    def metadata_processor_path(self) -> Path:
        return self.output_dir / "metadata_processor.joblib"

    @property
    def ensemble_prediction_path(self) -> Path:
        return self.prediction_dir / "ensemble.csv"

    def get_backbone_checkpoint_dir(self, backbone_name: str) -> Path:
        clean_name = backbone_name.replace("tf_", "").replace("-", "_")
        path = self.checkpoint_dir / clean_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Config:
        """Load Config dataclass from a YAML file."""
        import yaml
        from dataclasses import fields

        yaml_path = Path(yaml_path)
        config = cls()
        if not yaml_path.exists():
            print(f"[WARN] Config file not found: {yaml_path}, using defaults")
            return config

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in fields(cls)}

        for key, value in data.items():
            if key in valid_fields:
                setattr(config, key, value)
            if key == "backbone_name":
                config.model_name = value

        return config

