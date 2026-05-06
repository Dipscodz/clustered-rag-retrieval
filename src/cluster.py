"""Clustering utilities for clustered RAG retrieval."""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


def cluster_embeddings(embeddings_path: str = "embeddings/embeddings.npy", n_clusters: int = 10, model_path: str = "models/kmeans.npy") -> None:
    embeddings = np.load(embeddings_path)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(embeddings)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(model_path, np.array([model.cluster_centers_], dtype=object))
    print(f"Saved cluster centroids to {model_path}")


def load_cluster_centers(path: str = "models/kmeans.npy") -> Optional[np.ndarray]:
    if not Path(path).exists():
        return None
    return np.load(path, allow_pickle=True)
