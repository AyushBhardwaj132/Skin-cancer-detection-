from __future__ import annotations

import os
import json
import datetime
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class TrainingState:
    """Manages persistent training pipeline status for auto-resume, metric tracking, and checkpoint status."""
    completed_folds: list[int] = field(default_factory=list)
    current_fold: int = 0
    last_epoch: int = 0
    last_batch_idx: int = 0
    best_pauc: float = 0.0
    fold_best_scores: dict[str, float] = field(default_factory=dict)
    fold_checkpoints: dict[str, dict[str, str]] = field(default_factory=dict)
    hf_uploads: dict[str, str] = field(default_factory=dict)
    last_updated: str = ""

    def to_dict(self) -> dict:
        best_pauc_val = self.best_pauc
        if isinstance(best_pauc_val, float):
            if np.isnan(best_pauc_val) or best_pauc_val == float("-inf"):
                best_pauc_val = 0.0
            else:
                best_pauc_val = round(best_pauc_val, 4)

        clean_scores = {}
        for k, v in self.fold_best_scores.items():
            if isinstance(v, (float, np.floating)):
                clean_scores[str(k)] = round(float(v), 4) if not np.isnan(v) else 0.0
            else:
                clean_scores[str(k)] = v

        return {
            "completed_folds": sorted(list(set(self.completed_folds))),
            "current_fold": int(self.current_fold),
            "last_epoch": int(self.last_epoch),
            "last_batch_idx": int(self.last_batch_idx),
            "best_pauc": best_pauc_val,
            "fold_best_scores": clean_scores,
            "fold_checkpoints": self.fold_checkpoints,
            "hf_uploads": self.hf_uploads,
            "last_updated": datetime.datetime.now().isoformat(),
        }

    def update_epoch(
        self,
        fold: int,
        epoch: int,
        best_pauc: float = 0.0,
        batch_idx: int = 0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.current_fold = fold
        self.last_epoch = epoch
        self.last_batch_idx = batch_idx
        if best_pauc != float("-inf") and not np.isnan(best_pauc):
            self.best_pauc = best_pauc
            self.fold_best_scores[str(fold)] = best_pauc

        if checkpoint_path:
            fold_key = str(fold)
            if fold_key not in self.fold_checkpoints:
                self.fold_checkpoints[fold_key] = {}
            self.fold_checkpoints[fold_key]["last"] = str(checkpoint_path)

    def mark_fold_completed(self, fold: int, best_pauc: float, best_ckpt_path: str | Path) -> None:
        if fold not in self.completed_folds:
            self.completed_folds.append(fold)
        self.completed_folds = sorted(list(set(self.completed_folds)))
        self.fold_best_scores[str(fold)] = best_pauc
        fold_key = str(fold)
        if fold_key not in self.fold_checkpoints:
            self.fold_checkpoints[fold_key] = {}
        self.fold_checkpoints[fold_key]["best"] = str(best_ckpt_path)

    def record_hf_upload(self, fold: int, status_str: str) -> None:
        self.hf_uploads[str(fold)] = status_str

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
                fold_best_scores=data.get("fold_best_scores", {}),
                fold_checkpoints=data.get("fold_checkpoints", {}),
                hf_uploads=data.get("hf_uploads", {}),
                last_updated=data.get("last_updated", ""),
            )
        except Exception as e:
            print(f"[WARN] Existing training_state.json could not be parsed ({e}). Creating new state.")
            return cls()

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        state_path = output_path / "training_state.json"

        tmp_path = state_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(state_path)
        except Exception as e:
            raise RuntimeError(f"[FAIL FAST] Failed to write training_state.json: {e}")

        return state_path
