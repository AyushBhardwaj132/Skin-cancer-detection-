from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average (EMA) of model weights for enhanced validation performance & generalization."""
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str | torch.device | None = None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(self.device)

        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k].to(v.device) * (1.0 - self.decay))
                else:
                    v.copy_(msd[k].to(v.device))
