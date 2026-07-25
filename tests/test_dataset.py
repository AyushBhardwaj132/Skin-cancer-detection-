from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.dataset import ISICDataset


def test_isic_dataset_initialization():
    df = pd.DataFrame({
        "isic_id": ["ISIC_0000001", "ISIC_0000002"],
        "target": [0, 1],
    })
    meta_tensor = np.zeros((2, 10), dtype=np.float32)
    dataset = ISICDataset(df, image_dir="data", is_test=False, metadata_tensor=meta_tensor)

    assert len(dataset) == 2
    sample = dataset[0]
    assert "image" in sample
    assert "metadata" in sample
    assert "target" in sample
    assert sample["image"].shape == (3, 384, 384)
    assert sample["metadata"].shape == (10,)
    assert sample["target"] == 0.0
