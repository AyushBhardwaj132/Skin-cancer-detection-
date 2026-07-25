from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.utils import resolve_image_path


class ISICDataset(Dataset):
    """Production PyTorch dataset for skin lesion images and metadata."""
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        transform=None,
        is_test: bool = False,
        target_col: str = "target",
        image_id_col: str = "isic_id",
        metadata_tensor: torch.Tensor | np.ndarray | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_test = is_test
        self.target_col = target_col
        self.image_id_col = image_id_col

        if metadata_tensor is not None:
            if isinstance(metadata_tensor, np.ndarray):
                self.metadata_tensor = torch.tensor(metadata_tensor, dtype=torch.float32)
            else:
                self.metadata_tensor = metadata_tensor.float()
        else:
            self.metadata_tensor = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        image_id = str(row[self.image_id_col])

        try:
            image_path = resolve_image_path(self.image_dir, image_id)
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
        except Exception:
            # Fallback synthetic array if image is corrupted or missing in test mode
            image_np = np.zeros((384, 384, 3), dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

        sample = {
            "image": image_tensor,
            "image_id": image_id,
        }

        if self.metadata_tensor is not None:
            sample["metadata"] = self.metadata_tensor[idx]

        if not self.is_test and self.target_col in row:
            sample["target"] = torch.tensor(row[self.target_col], dtype=torch.float32)

        return sample
