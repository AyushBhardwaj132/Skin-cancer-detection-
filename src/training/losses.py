from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Numerically stable Focal Loss for severe class imbalance."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clamp logits to prevent FP16 overflow
        logits = torch.clamp(logits, min=-30.0, max=30.0)
        probs = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        loss = bce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """Numerically stable Asymmetric Loss (ASL) for severe class imbalance."""
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-7,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clamp logits to prevent FP16 overflow
        logits = torch.clamp(logits, min=-30.0, max=30.0)
        targets = targets.float()
        probs = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)

        p_pos = probs
        p_neg = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(min=self.eps, max=1.0)

        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps)) * ((1.0 - p_pos) ** self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps)) * (p_pos ** self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PolyLoss(nn.Module):
    """Numerically stable PolyLoss (Poly-1)."""
    def __init__(self, epsilon: float = 2.0, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, min=-30.0, max=30.0)
        probs = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        poly_loss = bce_loss + self.epsilon * (1.0 - p_t)

        if self.reduction == "mean":
            return poly_loss.mean()
        elif self.reduction == "sum":
            return poly_loss.sum()
        return poly_loss


def get_loss_fn(
    loss_type: str = "focal",
    alpha: float = 0.75,
    gamma: float = 2.0,
    pos_weight: float | None = None,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip_margin: float = 0.05,
    poly_epsilon: float = 2.0,
) -> nn.Module:
    """Factory builder for loss functions."""
    loss_type = loss_type.lower()
    if loss_type == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type in ("asymmetric", "asl"):
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip_margin)
    elif loss_type in ("polyloss", "poly"):
        return PolyLoss(epsilon=poly_epsilon)
    elif loss_type == "weighted_bce" and pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    else:
        return nn.BCEWithLogitsLoss()


build_loss = get_loss_fn
