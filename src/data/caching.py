from __future__ import annotations

import os
import hashlib
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from src.config.config import Config
from src.data.patient_features import enrich_metadata
from src.data.metadata import MetadataProcessor


class MetadataCacheManager:
    """Caches patient feature calculations and normalized metadata arrays to eliminate startup latency."""

    @staticmethod
    def _compute_config_hash(config: Config) -> str:
        """Creates a unique hash based on metadata configuration flags."""
        config_str = (
            f"use_patient_features={config.use_patient_features}_"
            f"use_ugly_duckling={config.use_ugly_duckling}_"
            f"mlp_hidden={config.metadata_mlp_hidden}_"
            f"mlp_output={config.metadata_mlp_output}"
        )
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

    @classmethod
    def get_cache_path(cls, metadata_path: Path, config: Config) -> Path:
        cache_dir = config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        config_hash = cls._compute_config_hash(config)
        return cache_dir / f"enriched_{metadata_path.stem}_{config_hash}.joblib"

    @classmethod
    def is_cache_valid(cls, metadata_path: Path, cache_path: Path) -> bool:
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            return False
        # Cache is invalid if original CSV file was modified after cache was created
        if metadata_path.exists() and metadata_path.stat().st_mtime > cache_path.stat().st_mtime:
            return False
        return True

    @classmethod
    def load_or_compute_enriched_metadata(
        cls,
        metadata_path: Path,
        config: Config,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Loads enriched metadata from cache or computes it if missing/stale."""
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        cache_path = cls.get_cache_path(metadata_path, config)

        if cls.is_cache_valid(metadata_path, cache_path):
            if verbose:
                print(f"  [FAST STARTUP] Loaded cached enriched metadata from {cache_path.name}")
            try:
                return joblib.load(cache_path)
            except Exception as e:
                if verbose:
                    print(f"  [CACHE WARN] Cache load failed ({e}), recomputing...")

        if verbose:
            print("  Computing patient features & ugly duckling scores (caching result)...")
        
        df = pd.read_csv(metadata_path)
        if config.use_patient_features:
            df = enrich_metadata(df)

        try:
            joblib.dump(df, cache_path)
            if verbose:
                print(f"  [CACHE] Saved enriched metadata to {cache_path}")
        except Exception as e:
            if verbose:
                print(f"  [CACHE WARN] Failed to save cache: {e}")

        return df
