from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

import io
import os
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
    """Production PyTorch dataset supporting extracted image folders and Kaggle HDF5 datasets."""
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

        self.image_ids = self.df[self.image_id_col].astype(str).values
        if not self.is_test and self.target_col in self.df.columns:
            self.targets = self.df[self.target_col].values.astype(np.float32)
        else:
            self.targets = None

        # --- HDF5 Auto Detection ---
        self.hdf5_path = self._detect_hdf5_path(self.image_dir)
        self._h5_file = None

        if metadata_tensor is not None:
            if isinstance(metadata_tensor, np.ndarray):
                self.metadata_tensor = torch.tensor(metadata_tensor, dtype=torch.float32)
            else:
                self.metadata_tensor = metadata_tensor.float()
        else:
            self.metadata_tensor = None

    @staticmethod
    def _detect_hdf5_path(image_dir: Path) -> Path | None:
        """Detects whether an HDF5 dataset exists locally or in Kaggle directories."""
        candidates = [
            image_dir if str(image_dir).endswith((".hdf5", ".h5")) and image_dir.is_file() else None,
            image_dir / "train-image.hdf5",
            image_dir / "test-image.hdf5",
            image_dir.parent / f"{image_dir.name}.hdf5",
            image_dir.with_suffix(".hdf5") if image_dir.suffix != ".hdf5" else None,
            Path("/kaggle/input/isic-2024-challenge/train-image.hdf5"),
            Path("/kaggle/input/isic-2024-challenge/test-image.hdf5"),
            Path("data/train-image.hdf5"),
            Path("data/test-image.hdf5"),
        ]
        for c in candidates:
            if c is not None and c.exists() and c.is_file():
                return c
        return None

    def _get_h5_file(self):
        """Lazy worker-safe h5py File handle retrieval."""
        if self._h5_file is None and self.hdf5_path is not None:
            try:
                import h5py
                self._h5_file = h5py.File(str(self.hdf5_path), "r")
            except Exception:
                self._h5_file = None
        return self._h5_file

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_id = self.image_ids[idx]
        image_np = None

        # 1. Try reading from HDF5 dataset if available
        if self.hdf5_path is not None:
            try:
                h5_file = self._get_h5_file()
                if h5_file is not None and image_id in h5_file:
                    raw_data = h5_file[image_id][()]
                    if isinstance(raw_data, bytes):
                        buf = np.frombuffer(raw_data, dtype=np.uint8)
                    elif isinstance(raw_data, np.ndarray):
                        buf = raw_data.astype(np.uint8)
                    else:
                        buf = np.frombuffer(bytes(raw_data), dtype=np.uint8)

                    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                image_np = None

        # 2. Try reading from extracted image directory if HDF5 fails or not present
        if image_np is None:
            try:
                image_path = resolve_image_path(self.image_dir, image_id)
                img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    image = Image.open(image_path).convert("RGB")
                    image_np = np.array(image)
            except Exception:
                # 3. Fallback synthetic array if image is corrupted or missing in test mode
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

        if self.targets is not None:
            sample["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return sample
