"""Preprocessing utilities for clustered RAG retrieval."""

from pathlib import Path
from typing import List


def load_data(path: str) -> List[str]:
    data_path = Path(path)
    if not data_path.exists():
        return []
    with data_path.open("r", encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip()]


def preprocess_texts(source_path: str = "data/raw.txt", target_path: str = "data/preprocessed.txt") -> None:
    texts = load_data(source_path)
    processed = [text.lower().strip() for text in texts]
    Path(target_path).write_text("\n".join(processed), encoding="utf-8")
    print(f"Preprocessed {len(processed)} texts into {target_path}")


def preprocess_text(text: str) -> str:
    return text.lower().strip()
