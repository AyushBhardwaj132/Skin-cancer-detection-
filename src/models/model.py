from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """Generalized Mean (GeM) Pooling layer for fine-grained image feature extraction."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def build_model(model_name: str = "efficientnetv2_m", pretrained: bool = True, num_classes: int = 1):
    candidate_names = [model_name, "efficientnetv2_m", "tf_efficientnetv2_m"]
    last_error: Exception | None = None
    for candidate in candidate_names:
        try:
            return timm.create_model(candidate, pretrained=pretrained, num_classes=num_classes)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to create a timm model for {model_name}: {last_error}")


def build_backbone(model_name: str = "tf_efficientnetv2_m", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a timm backbone for feature extraction."""
    candidate_names = [model_name, f"tf_{model_name}"]
    last_error: Exception | None = None
    for candidate in candidate_names:
        try:
            model = timm.create_model(candidate, pretrained=pretrained, num_classes=0)
            return model, model.num_features
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to create a timm backbone for {model_name}: {last_error}")


class ISICModel(nn.Module):
    def __init__(self, model_name: str = "efficientnetv2_m", pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        self.backbone = build_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)
