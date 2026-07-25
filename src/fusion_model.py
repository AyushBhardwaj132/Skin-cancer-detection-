from __future__ import annotations

import torch
import torch.nn as nn
from src.model import build_backbone

class MetadataMLP(nn.Module):
    """Small MLP to encode metadata features into a dense vector."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FusionModel(nn.Module):
    """Multimodal model that fuses image embeddings with metadata features."""
    def __init__(
        self,
        backbone_name: str = "tf_efficientnetv2_m",
        metadata_dim: int = 50,
        pretrained: bool = True,
        metadata_hidden: int = 256,
        metadata_output: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone, embed_dim = build_backbone(backbone_name, pretrained)
        self.metadata_mlp = MetadataMLP(metadata_dim, metadata_hidden, metadata_output, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + metadata_output, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
    
    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_features = self.backbone(images)           # (B, embed_dim)
        meta_features = self.metadata_mlp(metadata)    # (B, metadata_output)
        fused = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(fused)                  # (B, 1)
    
    def get_embeddings(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Get fused embeddings before the classifier (for t-SNE/UMAP)."""
        img_features = self.backbone(images)
        meta_features = self.metadata_mlp(metadata)
        return torch.cat([img_features, meta_features], dim=1)
