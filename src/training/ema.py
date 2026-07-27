from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average (EMA) of model weights with zero-allocation in-place tensor updates."""
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str | torch.device | None = None):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(self.device)

        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for ema_param, model_param in zip(self.module.parameters(), model.parameters()):
                if ema_param.dtype.is_floating_point:
                    ema_param.mul_(self.decay).add_(model_param.detach(), alpha=1.0 - self.decay)

            for ema_buf, model_buf in zip(self.module.buffers(), model.buffers()):
                ema_buf.copy_(model_buf.detach())
