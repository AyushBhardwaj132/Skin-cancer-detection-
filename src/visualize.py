from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE


def extract_embeddings(
    model,
    dataloader,
    device: torch.device,
    max_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fused embeddings from the model.
    
    The model must have a `get_embeddings(images, metadata)` method.
    The dataloader must yield (images, metadata, labels).
    
    Returns:
        (embeddings, labels) numpy arrays.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    total = 0
    with torch.no_grad():
        for images, metadata, labels in dataloader:
            if total >= max_samples:
                break
            images = images.to(device)
            metadata = metadata.to(device)
            emb = model.get_embeddings(images, metadata)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())
            total += images.size(0)
    return np.concatenate(all_embeddings)[:max_samples], np.concatenate(all_labels)[:max_samples]


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> None:
    """Create and save a 2D t-SNE plot colored by malignant/benign."""
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    coords = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    benign_mask = labels == 0
    malignant_mask = labels == 1
    
    ax.scatter(coords[benign_mask, 0], coords[benign_mask, 1], c='#4A90D9', alpha=0.4, s=10, label='Benign')
    ax.scatter(coords[malignant_mask, 0], coords[malignant_mask, 1], c='#E74C3C', alpha=0.8, s=25, label='Malignant', marker='x')
    
    ax.set_title('t-SNE Embedding Visualization', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved t-SNE plot to {save_path}')


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """Create and save a 2D UMAP plot colored by malignant/benign."""
    try:
        import umap
    except ImportError:
        print('umap-learn not installed, skipping UMAP visualization.')
        return
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    benign_mask = labels == 0
    malignant_mask = labels == 1
    
    ax.scatter(coords[benign_mask, 0], coords[benign_mask, 1], c='#4A90D9', alpha=0.4, s=10, label='Benign')
    ax.scatter(coords[malignant_mask, 0], coords[malignant_mask, 1], c='#E74C3C', alpha=0.8, s=25, label='Malignant', marker='x')
    
    ax.set_title('UMAP Embedding Visualization', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved UMAP plot to {save_path}')


def visualize_embeddings(
    model,
    dataloader,
    device: torch.device,
    output_dir: str | Path,
    fold_idx: int = 0,
    max_samples: int = 5000,
) -> None:
    """Full pipeline: extract embeddings → plot t-SNE and UMAP → save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Extracting embeddings (max {max_samples} samples)...')
    embeddings, labels = extract_embeddings(model, dataloader, device, max_samples)
    print(f'Got {len(embeddings)} embeddings of dim {embeddings.shape[1]}')
    
    # Filter valid labels
    valid = labels >= 0
    embeddings = embeddings[valid]
    labels = labels[valid]
    
    plot_tsne(embeddings, labels, output_dir / f'tsne_fold{fold_idx}.png')
    plot_umap(embeddings, labels, output_dir / f'umap_fold{fold_idx}.png')
