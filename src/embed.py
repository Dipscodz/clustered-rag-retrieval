"""Embedding utilities for clustered RAG retrieval."""

from pathlib import Path
from typing import List

import numpy as np


def load_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required for embedding") from exc
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = load_embeddings_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.asarray(embeddings, dtype=np.float32)


def save_embeddings(embeddings: np.ndarray, path: str = "embeddings/embeddings.npy") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
