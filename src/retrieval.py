from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class LesionRetrievalIndex:
    """Nearest-Neighbor similarity search index over 2048-D lesion embeddings."""
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.nn_model = NearestNeighbors(n_neighbors=10, metric=metric)
        self.embeddings = None
        self.metadata_df = None
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray, metadata_df: pd.DataFrame):
        self.embeddings = embeddings.astype(np.float32)
        self.metadata_df = metadata_df.reset_index(drop=True).copy()
        self.nn_model.fit(self.embeddings)
        self.is_fitted = True
        print(f"Indexed {len(embeddings)} lesion embeddings (dim={embeddings.shape[1]}).")
        return self

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """Search for top k visually similar lesions given a 1D or 2D query embedding tensor."""
        if not self.is_fitted:
            raise RuntimeError("LesionRetrievalIndex must be fitted before running search.")
            
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.nn_model.kneighbors(query_embedding.astype(np.float32), n_neighbors=k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.metadata_df.iloc[idx].to_dict()
            results.append({
                "isic_id": str(row.get("isic_id", f"sample_{idx}")),
                "target": int(row.get("target", 0)) if "target" in row and not pd.isna(row["target"]) else None,
                "distance": float(dist),
                "similarity_score": float(1.0 - dist) if self.metric == "cosine" else float(1.0 / (1.0 + dist)),
                "age": row.get("age_approx", "N/A"),
                "anatom_site": row.get("anatom_site_general", "N/A"),
            })
        return results

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.nn_model, "embeddings": self.embeddings, "metadata": self.metadata_df, "metric": self.metric}, path)
        print(f"Saved retrieval index to {path}")

    @classmethod
    def load(cls, path: str | Path) -> LesionRetrievalIndex:
        data = joblib.load(path)
        obj = cls(metric=data["metric"])
        obj.nn_model = data["model"]
        obj.embeddings = data["embeddings"]
        obj.metadata_df = data["metadata"]
        obj.is_fitted = True
        return obj
