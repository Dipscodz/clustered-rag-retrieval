"""Generation utilities for clustered RAG retrieval."""

from typing import List


def generate_answer(query: str, context: List[str] = None) -> str:
    context = context or []
    joined = "\n".join(context)
    return f"Generated answer for query: {query}\nContext:\n{joined}"


def rank_candidates(candidates: List[str]) -> List[str]:
    return sorted(candidates, key=len)
