"""Evaluation utilities for clustered RAG retrieval."""

from typing import List


def evaluate_retrieval(predictions: List[str] = None, references: List[str] = None) -> None:
    predictions = predictions or []
    references = references or []
    matches = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    accuracy = matches / len(references) if references else 0.0
    print(f"Retrieval accuracy: {accuracy:.2%} ({matches}/{len(references)})")


def evaluate_clustering(labels: List[int] = None, references: List[int] = None) -> None:
    labels = labels or []
    references = references or []
    if not labels or not references or len(labels) != len(references):
        print("Clustering evaluation requires equal-length label lists")
        return
    matches = sum(1 for a, b in zip(labels, references) if a == b)
    print(f"Clustering match rate: {matches / len(labels):.2%}")
