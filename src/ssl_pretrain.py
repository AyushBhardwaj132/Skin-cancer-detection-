from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import build_backbone


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR contrastive loss)."""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Args:
            z_i, z_j: Normalized projections of two augmented views of same batch (B, D).
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        z = F.normalize(z, dim=1)
        
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        
        # Mask out self-contrast
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Positive targets: i -> i+B and i+B -> i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ])
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class SimCLRModel(nn.Module):
    """Self-supervised SimCLR model with projection head for unlabelled lesion pretraining."""
    def __init__(self, backbone_name: str = "tf_efficientnetv2_m", projection_dim: int = 128):
        super().__init__()
        self.backbone, in_features = build_backbone(backbone_name, pretrained=True)
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)
