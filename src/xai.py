from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM visual explanation generator for CNN visual backbones."""
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer(model)
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Find the last convolutional layer in the backbone."""
        target = None
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                target = module
        if target is None:
            raise ValueError("Could not automatically locate a Conv2d target layer for Grad-CAM.")
        return target

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        metadata_tensor: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Generate Grad-CAM activation heatmap for a single image tensor (1, C, H, W)."""
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)
        
        if metadata_tensor is not None and metadata_tensor.dim() == 1:
            metadata_tensor = metadata_tensor.unsqueeze(0)

        # Forward pass
        if metadata_tensor is not None and hasattr(self.model, "metadata_mlp"):
            logits = self.model(image_tensor, metadata_tensor)
        else:
            logits = self.model(image_tensor)

        # Backward pass on predicted output score
        score = logits[0, 0]
        self.model.zero_grad()
        score.backward()

        # Compute neuron importance weights via global average pooling of gradients
        gradients = self.gradients.detach().cpu().numpy()[0]     # (C, h, w)
        activations = self.activations.detach().cpu().numpy()[0] # (C, h, w)
        weights = np.mean(gradients, axis=(1, 2))                # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to keep positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        # Resize to match input image spatial resolution (H, W)
        orig_h, orig_w = image_tensor.size(2), image_tensor.size(3)
        cam = cv2.resize(cam, (orig_w, orig_h))
        return cam


def overlay_heatmap_on_image(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image array (H, W, 3)."""
    # Ensure image_np is uint8 in range [0, 255]
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
        
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 1 - alpha, color_heatmap, alpha, 0)
    return overlay


def save_gradcam_visualization(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    save_path: str | Path,
    title: str = "Grad-CAM Explanation",
) -> None:
    """Save original image, heatmap, and overlay visualization side-by-side."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = overlay_heatmap_on_image(image_np, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Lesion Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Explanation Overlay")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved Grad-CAM visualization to {save_path}")
