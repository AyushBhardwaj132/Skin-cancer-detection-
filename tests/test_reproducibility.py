from __future__ import annotations

import random
import numpy as np
import torch

from src.utils import set_seed, seed_worker


def test_seed_everything_deterministic():
    set_seed(42)
    val1 = random.random()
    np_val1 = np.random.rand(5)
    torch_val1 = torch.randn(5)

    set_seed(42)
    val2 = random.random()
    np_val2 = np.random.rand(5)
    torch_val2 = torch.randn(5)

    assert val1 == val2
    assert np.allclose(np_val1, np_val2)
    assert torch.allclose(torch_val1, torch_val2)


def test_seed_worker_execution():
    # Verify worker initialization function does not raise
    seed_worker(0)
    seed_worker(1)
