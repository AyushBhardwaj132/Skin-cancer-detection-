from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_term = alpha_t * ((1 - pt) ** self.gamma)
        
        loss = focal_term * bce
        return loss.mean()

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss — different focusing for positives vs negatives.
    
    Positive samples: standard focal with gamma_pos
    Negative samples: harder focusing with gamma_neg + probability shifting
    """
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        p_pos = probs
        p_neg = 1 - probs
        
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)
            
        loss_pos = -targets * ((1 - p_pos) ** self.gamma_pos) * torch.log(p_pos.clamp(min=1e-8))
        loss_neg = -(1 - targets) * ((1 - p_neg) ** self.gamma_neg) * torch.log(p_neg.clamp(min=1e-8))
        
        loss = loss_pos + loss_neg
        return loss.mean()

def build_loss(loss_type: str = "bce", **kwargs) -> nn.Module:
    """Factory function for loss functions."""
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "asymmetric":
        return AsymmetricLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
