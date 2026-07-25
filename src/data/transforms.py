from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool = True, image_size: int = 384, use_advanced: bool = False) -> A.Compose:
    """Build Albumentations transform pipeline with full v2.0+ API compliance."""
    if train:
        if use_advanced:
            return A.Compose([
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_REFLECT),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
                A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(int(image_size * 0.05), int(image_size * 0.1)), hole_width_range=(int(image_size * 0.05), int(image_size * 0.1)), p=0.2),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.1),
                A.CLAHE(clip_limit=(1.0, 2.0), p=0.2),
                A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(int(image_size * 0.02), int(image_size * 0.05)), hole_width_range=(int(image_size * 0.02), int(image_size * 0.05)), p=0.3),
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_REFLECT),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
                A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(int(image_size * 0.05), int(image_size * 0.1)), hole_width_range=(int(image_size * 0.05), int(image_size * 0.1)), p=0.2),
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def build_tta_transforms(image_size: int = 384) -> list[A.Compose]:
    """Return a list of deterministic transformations for Test-Time Augmentation (TTA)."""
    return [
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
    ]


def mixup_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation to a batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def cutmix_data(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation to a batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    
    H, W = images.size(2), images.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_images = images.clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return mixed_images, labels, labels[index], lam
