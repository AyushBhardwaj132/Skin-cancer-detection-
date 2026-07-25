from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.fusion_model import FusionModel
from src.utils import get_device


class DistillationLoss(nn.Module):
    """Knowledge Distillation loss combining soft teacher loss and hard ground-truth loss.
    
    L_distill = alpha * (T^2 * KL(sigmoid(z_s/T), sigmoid(z_t/T))) + (1 - alpha) * BCE(z_s, y)
    """
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Soft loss via KL divergence on scaled probabilities
        t = self.temperature
        p_student = torch.sigmoid(student_logits / t)
        p_teacher = torch.sigmoid(teacher_logits / t)
        
        # Binary KL divergence: p_t * log(p_t/p_s) + (1-p_t) * log((1-p_t)/(1-p_s))
        kl_soft = p_teacher * torch.log((p_teacher + 1e-7) / (p_student + 1e-7)) + \
                  (1 - p_teacher) * torch.log((1 - p_teacher + 1e-7) / (1 - p_student + 1e-7))
        soft_loss = kl_soft.mean() * (t ** 2)
        
        # Hard loss with true labels
        hard_loss = self.bce(student_logits, targets)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def train_student_model(
    teacher_model: nn.Module,
    train_loader,
    val_loader,
    metadata_dim: int,
    student_backbone: str = "tf_efficientnetv2_s",
    epochs: int = 5,
    learning_rate: float = 1e-4,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> nn.Module:
    """Train a small student model using distillation from a heavy teacher ensemble."""
    device = get_device()
    teacher_model.eval().to(device)
    
    student_model = FusionModel(
        backbone_name=student_backbone,
        metadata_dim=metadata_dim,
        pretrained=True,
    ).to(device)
    
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = AdamW(student_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    print(f"Distilling knowledge into student model '{student_backbone}' (T={temperature}, alpha={alpha})...")
    
    for epoch in range(1, epochs + 1):
        student_model.train()
        running_loss = 0.0
        
        for images, metadata, labels in tqdm(train_loader, desc=f"Student Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            with torch.no_grad():
                teacher_logits = teacher_model(images, metadata)
                
            optimizer.zero_grad()
            student_logits = student_model(images, metadata)
            loss = criterion(student_logits, teacher_logits, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Student Epoch {epoch}/{epochs} — Loss: {epoch_loss:.4f}")
        
    return student_model
