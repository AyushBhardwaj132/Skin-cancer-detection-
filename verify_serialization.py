"""Minimal verification: Config serialization with dataclasses.asdict works."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import asdict
from src.config import Config
from src.utils import save_checkpoint, ensure_dir

# 1. Load config from YAML
config = Config.from_yaml("configs/kaggle_config.yaml")
print("[OK] Config.from_yaml() succeeded")

# 2. Serialize with asdict (the line that previously crashed)
config_dict = {
    k: str(v) if isinstance(v, Path) else v
    for k, v in asdict(config).items()
}
print(f"[OK] asdict(config) succeeded — {len(config_dict)} fields serialized")

# 3. Build a mock checkpoint payload (same structure as src/train.py L413-427)
checkpoint_payload = {
    "epoch": 1,
    "fold": 0,
    "model_name": config.backbone_name,
    "model_state_dict": {},
    "optimizer_state_dict": {},
    "scheduler_state_dict": {},
    "scaler_state_dict": None,
    "ema_state_dict": None,
    "best_val_pauc": 0.123,
    "best_val_auc": 0.456,
    "metadata_dim": 55,
    "use_metadata": config.use_metadata,
    "config": config_dict,
}

# 4. Save checkpoint to a temp directory
tmp_dir = Path("outputs/_verify_serialization")
ensure_dir(tmp_dir)
ckpt_path = tmp_dir / "test_checkpoint.pt"
save_checkpoint(checkpoint_payload, ckpt_path)
print(f"[OK] Checkpoint saved to {ckpt_path} ({ckpt_path.stat().st_size} bytes)")

# 5. Reload and verify
from src.utils import load_checkpoint
loaded = load_checkpoint(ckpt_path)
loaded_cfg = loaded["config"]
print(f"[OK] Checkpoint reloaded — config has {len(loaded_cfg)} fields")
print(f"     backbone_name = {loaded_cfg['backbone_name']}")
print(f"     image_size    = {loaded_cfg['image_size']}")
print(f"     focal_alpha   = {loaded_cfg['focal_alpha']}")

# 6. Cleanup
ckpt_path.unlink()
tmp_dir.rmdir()
print("[OK] Cleanup done")
print("\n[PASS] ALL SERIALIZATION CHECKS PASSED — no AttributeError")
