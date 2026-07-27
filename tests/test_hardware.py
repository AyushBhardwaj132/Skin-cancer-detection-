import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn as nn
from src.training.hardware import get_hardware_info, setup_accelerated_model


def test_get_hardware_info():
    hw = get_hardware_info()
    assert "is_cuda" in hw
    assert "gpu_count" in hw
    assert "device_name" in hw
    assert "is_ddp" in hw


def test_setup_accelerated_model():
    device = torch.device("cpu")
    model = nn.Linear(10, 1)
    
    sample_batch = {
        "image": torch.randn(4, 3, 224, 224),
        "metadata": torch.randn(4, 10),
    }

    acc_model, metrics = setup_accelerated_model(model, device, sample_batch=sample_batch)
    assert acc_model is not None
    assert "mode" in metrics
