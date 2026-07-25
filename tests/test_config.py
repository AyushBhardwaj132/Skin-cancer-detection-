from src.config import Config


def test_config_defaults():
    config = Config()
    assert config.project_name == "ISIC2024"
    assert config.image_size == 384
    assert config.batch_size == 32
    assert config.n_splits == 5
    assert config.loss_type == "focal"


def test_config_paths():
    config = Config()
    assert config.train_metadata_path.name == "train-metadata.csv"
    assert config.test_metadata_path.name == "test-metadata.csv"
    assert config.best_checkpoint_path.name == "best_model.pt"
