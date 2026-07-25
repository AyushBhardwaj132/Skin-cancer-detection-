from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for feature map explainability."""
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_target_layer()

        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        backbone = getattr(self.model, "backbone", self.model)
        conv_layers = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            raise RuntimeError("No Conv2d layers found in backbone model.")
        return conv_layers[-1]

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor, metadata_tensor: torch.Tensor | None = None) -> np.ndarray:
        self.model.zero_grad()
        if metadata_tensor is not None:
            output = self.model(image_tensor, metadata_tensor)
        else:
            output = self.model(image_tensor)

        score = output[0, 0]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed to capture gradients or activations.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_heatmap_on_image(image: Image.Image | np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlays Grad-CAM heatmap onto lesion image."""
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()

    h, w = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 1.0 - alpha, color_map, alpha, 0)
    return Image.fromarray(overlay)
