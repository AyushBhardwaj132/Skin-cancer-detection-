from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

import cv2
from src.utils import resolve_image_path


def crop_lesion_centered(img_np: np.ndarray, margin: float = 0.20) -> np.ndarray:
    """Detects lesion ROI and crops square region around lesion with margin."""
    h, w, _ = img_np.shape
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_np

    c_max = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c_max)
    if area < (h * w * 0.01) or area > (h * w * 0.95):
        return img_np

    x, y, bw, bh = cv2.boundingRect(c_max)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + margin)

    x1 = max(0, int(cx - side / 2.0))
    y1 = max(0, int(cy - side / 2.0))
    x2 = min(w, int(cx + side / 2.0))
    y2 = min(h, int(cy + side / 2.0))

    cropped = img_np[y1:y2, x1:x2]
    if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return img_np
    return cropped


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
        use_lesion_crop: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_test = is_test
        self.target_col = target_col
        self.image_id_col = image_id_col
        self.use_lesion_crop = use_lesion_crop

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

        if self.use_lesion_crop:
            image_np = crop_lesion_centered(image_np)

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
