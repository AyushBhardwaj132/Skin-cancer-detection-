from __future__ import annotations

import os
from pathlib import Path
from src.config.config import Config


def validate_config(config: Config) -> tuple[bool, list[str]]:
    """Validates configuration parameters and returns (is_valid, list_of_errors)."""
    errors: list[str] = []

    # 1. Cross Validation
    if config.n_splits < 2:
        errors.append(f"Invalid n_splits={config.n_splits}. Must be >= 2.")

    # 2. Hyperparameters
    if config.batch_size <= 0:
        errors.append(f"Invalid batch_size={config.batch_size}. Must be > 0.")

    if config.num_epochs <= 0:
        errors.append(f"Invalid num_epochs={config.num_epochs}. Must be > 0.")

    if config.learning_rate <= 0 or config.learning_rate >= 1.0:
        errors.append(f"Invalid learning_rate={config.learning_rate}. Must be between 0 and 1.")

    if config.weight_decay < 0:
        errors.append(f"Invalid weight_decay={config.weight_decay}. Must be >= 0.")

    # 3. Loss Types
    allowed_losses = {"focal", "asymmetric", "polyloss", "weighted_bce", "bce"}
    if config.loss_type not in allowed_losses:
        errors.append(f"Invalid loss_type='{config.loss_type}'. Must be one of {allowed_losses}.")

    # 4. Data Directory & Files
    if config.use_metadata:
        if config.metadata_mlp_hidden <= 0 or config.metadata_mlp_output <= 0:
            errors.append("Metadata MLP dimensions (hidden and output) must be > 0.")

    # 5. Hardware / Parallelism
    if config.num_workers < 0:
        errors.append(f"Invalid num_workers={config.num_workers}. Must be >= 0.")

    if config.gradient_accumulation_steps <= 0:
        errors.append(f"Invalid gradient_accumulation_steps={config.gradient_accumulation_steps}. Must be >= 1.")

    return len(errors) == 0, errors


def ensure_valid_config(config: Config) -> None:
    """Raises ValueError if configuration is invalid."""
    is_valid, errors = validate_config(config)
    if not is_valid:
        error_msg = "\n".join([f"  - {err}" for err in errors])
        raise ValueError(f"[CONFIG ERROR] Invalid Pipeline Configuration:\n{error_msg}")
