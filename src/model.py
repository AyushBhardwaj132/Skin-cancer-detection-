from __future__ import annotations

import torch.nn as nn
import timm


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
    """Build a timm backbone for feature extraction (no classifier head).
    
    Returns:
        (backbone_model, embed_dim) tuple.
    """
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

    def forward(self, inputs):
        return self.backbone(inputs)
