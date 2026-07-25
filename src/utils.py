from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_image_path(image_dir: str | Path, image_id: str) -> Path:
    base_dir = Path(image_dir)
    if base_dir.is_file():
        raise FileNotFoundError(f"Expected an image directory, got file: {base_dir}")

    search_roots = [base_dir, base_dir / "image"]
    for root in search_roots:
        for extension in IMAGE_EXTENSIONS:
            candidate = root / f"{image_id}{extension}"
            if candidate.exists():
                return candidate

    for root in search_roots:
        if root.exists():
            for candidate in root.glob(f"{image_id}.*"):
                if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                    return candidate

    raise FileNotFoundError(f"Unable to locate an image for {image_id} under {base_dir}")


def save_checkpoint(state: dict, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location)
