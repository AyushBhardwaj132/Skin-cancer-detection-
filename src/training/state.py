from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
import numpy as np


@dataclass
class TrainingState:
    completed_folds: list[int] = field(default_factory=list)
    current_fold: int = 0
    last_epoch: int = 0
    last_batch_idx: int = 0
    best_pauc: float = 0.0

    def to_dict(self) -> dict:
        best_pauc_val = self.best_pauc
        if isinstance(best_pauc_val, float):
            if np.isnan(best_pauc_val) or best_pauc_val == float("-inf"):
                best_pauc_val = 0.0
            else:
                best_pauc_val = round(best_pauc_val, 4)

        return {
            "completed_folds": sorted(list(set(self.completed_folds))),
            "current_fold": int(self.current_fold),
            "last_epoch": int(self.last_epoch),
            "last_batch_idx": int(self.last_batch_idx),
            "best_pauc": best_pauc_val,
        }

    def update_epoch(self, fold: int, epoch: int, best_pauc: float = 0.0, batch_idx: int = 0) -> None:
        self.current_fold = fold
        self.last_epoch = epoch
        self.last_batch_idx = batch_idx
        if best_pauc != float("-inf") and not np.isnan(best_pauc):
            self.best_pauc = best_pauc

    @classmethod
    def load(cls, output_dir: str | Path) -> TrainingState:
        state_path = Path(output_dir) / "training_state.json"
        if not state_path.exists():
            return cls()

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                completed_folds=data.get("completed_folds", []),
                current_fold=data.get("current_fold", 0),
                last_epoch=data.get("last_epoch", 0),
                last_batch_idx=data.get("last_batch_idx", 0),
                best_pauc=data.get("best_pauc", 0.0),
            )
        except Exception as e:
            print(f"[WARN] Failed to read training_state.json ({e}). Returning default state.", flush=True)
            return cls()

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        state_path = output_path / "training_state.json"

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

        print("✓ Resume information saved", flush=True)
        sys.stdout.flush()
        return state_path
