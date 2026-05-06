"""Retrieval utilities for clustered RAG retrieval."""

from pathlib import Path
from typing import List, Tuple

import numpy as np


def build_index(embeddings_path: str = "embeddings/embeddings.npy") -> None:
    embeddings = np.load(embeddings_path)
    Path("models").mkdir(parents=True, exist_ok=True)
    print(f"Index build placeholder for {embeddings.shape[0]} embeddings")


def retrieve_neighbors(query_embedding: np.ndarray, embeddings_path: str = "embeddings/embeddings.npy", top_k: int = 5) -> List[Tuple[int, float]]:
    embeddings = np.load(embeddings_path)
    scores = embeddings.dot(query_embedding)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(int(idx), float(scores[idx])) for idx in top_indices]
