from __future__ import annotations

import torch
import torch.nn as nn


class PatientAttentionNetwork(nn.Module):
    """Patient Attention Network that dynamically pools lesion embeddings per patient.
    
    Given a set of lesion feature vectors belonging to the same patient (B, N_lesions, embed_dim),
    multi-head self-attention computes query-key alignment weights to form a unified Patient Context Vector.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, lesion_embeddings: torch.Tensor) -> torch.Tensor:
        """Args:
            lesion_embeddings: Tensor of shape (B, N_lesions, embed_dim) or (N_lesions, embed_dim).
        
        Returns:
            Patient context vector of shape (B, embed_dim) or (1, embed_dim).
        """
        if lesion_embeddings.dim() == 2:
            lesion_embeddings = lesion_embeddings.unsqueeze(0)  # (1, N_lesions, D)

        # Self-attention over patient lesions
        attn_out, _ = self.attn(lesion_embeddings, lesion_embeddings, lesion_embeddings)
        x = self.norm(lesion_embeddings + attn_out)
        
        # Mean pool across lesions for global patient context
        patient_context = x.mean(dim=1)  # (B, D)
        patient_context = patient_context + self.mlp(patient_context)
        return patient_context
