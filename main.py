"""Entry point for the clustered RAG retrieval project."""

import argparse

from src.preprocess import preprocess_texts
from src.embed import embed_texts
from src.cluster import cluster_embeddings
from src.retrieve import build_index
from src.generate import generate_answer
from src.evaluate import evaluate_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustered RAG retrieval pipeline")
    parser.add_argument("command", choices=["preprocess", "embed", "cluster", "index", "generate", "evaluate"], help="Pipeline stage to run")
    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_texts()
    elif args.command == "embed":
        embed_texts()
    elif args.command == "cluster":
        cluster_embeddings()
    elif args.command == "index":
        build_index()
    elif args.command == "generate":
        generate_answer("Example query")
    elif args.command == "evaluate":
        evaluate_retrieval()


if __name__ == "__main__":
    main()
