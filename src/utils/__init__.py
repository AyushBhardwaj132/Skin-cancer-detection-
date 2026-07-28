from __future__ import annotations

import sys
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


def sync_file(path: str | Path) -> Path:
    """Flush and sync OS kernel page cache buffer to physical disk storage for a file."""
    path_obj = Path(path).resolve()
    if not os.path.exists(path_obj):
        raise RuntimeError(f"[CRITICAL FAILURE] Cannot sync file because it does not exist: {path_obj}")
    try:
        with open(path_obj, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass
    return path_obj


def save_checkpoint(state: dict, path: str | Path) -> Path:
    path = Path(path).resolve()
    ensure_dir(path.parent)

    # Task 4: Write to file handle, flush Python stdio buffer, call os.fsync on file descriptor
    try:
        with open(path, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        raise RuntimeError(f"[FAIL FAST] Failed to execute torch.save with flush/fsync to {path}: {e}")

    # Task 5: Physical existence check using os.path.exists
    if not os.path.exists(path):
        raise RuntimeError(f"[CRITICAL FAILURE] Checkpoint file DOES NOT EXIST on disk after save: {path}")

    # Verify size > 0
    size_bytes = os.path.getsize(path)
    if size_bytes == 0:
        raise RuntimeError(f"[CRITICAL FAILURE] Checkpoint file exists but is 0 bytes: {path}")

    # Task 8: Immediate torch.load reload verification
    try:
        _ = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as load_err:
        raise RuntimeError(f"[CRITICAL FAILURE] Checkpoint file corrupt or unreadable immediately after write: {path} (Error: {load_err})")

    # Task 2 formatted output
    size_mb = size_bytes / (1024 * 1024)
    print("\nCheckpoint saved:\n", flush=True)
    print(f"{path}\n", flush=True)
    print(f"Exists: {os.path.exists(path)}\n", flush=True)
    print(f"Size: {size_mb:.2f} MB ({size_bytes} bytes)\n", flush=True)
    print("Reload test: PASSED\n", flush=True)

    print("Directory contents:", flush=True)
    dir_files = sorted([f.name for f in path.parent.iterdir() if f.is_file()])
    if not dir_files:
        print("(Empty)", flush=True)
    else:
        for fname in dir_files:
            print(f"- {fname}", flush=True)
    print("\n", flush=True)
    sys.stdout.flush()

    return path


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> dict:
    """Loads a PyTorch model checkpoint.

    Args:
        path: Path to the checkpoint file (.pt or .pth).
        map_location: Target device for loaded tensors (default "cpu").
        weights_only: If True, uses PyTorch weights-only unpickler. Defaults to False
            to allow loading full state dictionaries, optimizers, schedulers, and configs
            without PyTorch 2.6+ unpickling errors.

    Returns:
        Loaded state dictionary.
    """
    path_obj = Path(path)
    try:
        return torch.load(path_obj, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Fallback for older PyTorch versions (<2.0) that do not accept weights_only kwarg
        return torch.load(path_obj, map_location=map_location)



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
    "sync_file",
    "IMAGE_EXTENSIONS",
]

