from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Config:
    project_name: str = "ISIC2024"
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    train_metadata_name: str = "train-metadata.csv"
    test_metadata_name: str = "test-metadata.csv"
    sample_submission_name: str = "sample_submission.csv"
    train_image_dir_name: str = "train-image"
    test_image_dir_name: str = "test-image"
    best_checkpoint_name: str = "best_model.pt"
    submission_name: str = "submission.csv"
    model_name: str = "tf_efficientnetv2_m"
    image_size: int = 384
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    val_size: float = 0.2
    seed: int = 42
    target_column: str = "target"
    image_id_column: str = "isic_id"
    # Phase 2: GroupKFold and Early Stopping
    n_splits: int = 5
    early_stopping_patience: int = 5
    pauc_max_fpr: float = 0.1
    # Phase 3: Metadata Fusion
    use_metadata: bool = True
    use_patient_features: bool = True
    use_ugly_duckling: bool = True
    metadata_mlp_hidden: int = 256
    metadata_mlp_output: int = 128
    # Phase 3: Loss function
    loss_type: str = "focal"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    # Phase 3: Advanced augmentations
    use_advanced_augs: bool = True
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    # Phase 3: Backbone
    backbone_name: str = "tf_efficientnetv2_m"
    # Phase 3: Visualization
    viz_max_samples: int = 5000
    # Phase 4: Ensemble & Competition Optimization
    ensemble_backbones: tuple = ("tf_efficientnetv2_l", "convnext_base", "swin_base_patch4_window12_384")
    use_tta: bool = True
    tta_steps: int = 4
    ensemble_method: str = "rank"  # simple, weighted, rank
    pseudo_pos_thresh: float = 0.99
    pseudo_neg_thresh: float = 0.01
    use_fp16: bool = True

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

