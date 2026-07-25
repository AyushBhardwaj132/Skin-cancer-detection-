from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.utils import resolve_image_path


class ISICDataset(Dataset):
    """ISIC 2024 dataset supporting image + optional metadata features.

    When ``metadata_features`` is provided, ``__getitem__`` returns a
    ``(image, metadata, label)`` triple.  Otherwise it returns
    ``(image, metadata_zeros, label)`` where the metadata tensor is all
    zeros — this keeps the collate interface consistent so the training
    loop never has to branch.
    """

    def __init__(
        self,
        metadata,
        image_dir: str | Path,
        transform=None,
        target_column: str | None = "target",
        image_id_column: str = "isic_id",
        metadata_features: np.ndarray | None = None,
    ) -> None:
        if isinstance(metadata, pd.DataFrame):
            self.metadata = metadata.reset_index(drop=True).copy()
        else:
            self.metadata = pd.read_csv(metadata).reset_index(drop=True)

        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_column = target_column
        self.image_id_column = image_id_column

        # Pre-computed metadata feature matrix (N, D) from MetadataProcessor
        if metadata_features is not None:
            self.metadata_features = torch.tensor(
                metadata_features, dtype=torch.float32,
            )
        else:
            self.metadata_features = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_id = str(row[self.image_id_column])
        image_path = resolve_image_path(self.image_dir, image_id)

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if self.transform is None:
            image_tensor = F.to_tensor(image_np / 255.0)
        else:
            transformed = self.transform(image=image_np)
            image_tensor = transformed["image"]

        # Metadata feature vector
        if self.metadata_features is not None:
            meta_tensor = self.metadata_features[index]
        else:
            meta_tensor = torch.zeros(1, dtype=torch.float32)

        # Label
        if self.target_column is None or self.target_column not in row.index:
            label_value = -1.0
        else:
            label_value = row[self.target_column]
            if pd.isna(label_value):
                label_value = -1.0

        label = torch.tensor(float(label_value), dtype=torch.float32)
        return image_tensor, meta_tensor, label
