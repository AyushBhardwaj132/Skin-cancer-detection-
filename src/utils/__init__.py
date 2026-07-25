from __future__ import annotations

import random
import os
from pathlib import Path
import numpy as np
import torch
from src.utils.logger import get_logger

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything = set_seed


def get_device() -> torch.device:
    """Return available torch device (cuda, mps, cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_image_path(image_dir: str | Path, image_id: str) -> Path:
    """Locates image file matching image_id within image_dir or subdirectories."""
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


def seed_worker(worker_id: int) -> None:
    """Ensures deterministic random seeding across multi-worker DataLoaders."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = [
    "get_logger",
    "set_seed",
    "seed_everything",
    "seed_worker",
    "get_device",
    "ensure_dir",
    "resolve_image_path",
    "save_checkpoint",
    "load_checkpoint",
    "IMAGE_EXTENSIONS",
]

