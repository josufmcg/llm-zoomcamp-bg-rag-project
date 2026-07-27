"""Run search evaluation metrics against ground truth data.

Evaluates all three search methods (vector, keyword, hybrid) using
Hit Rate@5 and MRR@5, and displays results in a table.

Usage:
    uv run python scripts/evaluate.py

Prerequisites:
    - PostgreSQL running with documents ingested
    - data/ground_truth.csv must exist (run scripts/generate_ground_truth.py first)
"""

import csv
from pathlib import Path

from dotenv import load_dotenv

from bg_rag.embedder import Embedder
from bg_rag.evaluation import evaluate_search
from bg_rag.search import SearchEngine


def load_ground_truth() -> list[dict]:
    """Load ground truth data from CSV."""
    path = Path("data/ground_truth.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {path}. "
            "Run 'uv run python scripts/generate_ground_truth.py' first."
        )

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    load_dotenv()

    print("Loading ground truth data...")
    ground_truth = load_ground_truth()
    print(f"  {len(ground_truth)} evaluation queries")

    print("Initializing search engine...")
    embedder = Embedder()
    search_engine = SearchEngine(embedder)

    methods = ["vector", "keyword", "hybrid"]
    k = 5

    print(f"\nEvaluating search methods (K={k}):")
    print(f"{'Method':<10} {'Hit Rate@5':>12} {'MRR@5':>12} {'Queries':>10}")
    print("-" * 48)

    results = {}
    for method in methods:
        def search_func(query, m=method):
            return search_engine.search(query, method=m, limit=k)

        metrics = evaluate_search(ground_truth, search_func, k=k)
        results[method] = metrics

        print(
            f"{method:<10} {metrics['hit_rate']:>12.4f} {metrics['mrr']:>12.4f} "
            f"{metrics['total_queries']:>10d}"
        )

    print("-" * 48)

    # Determine best method
    best_method = max(results.keys(), key=lambda m: results[m]["mrr"])
    print(f"\nBest method by MRR: {best_method} ({results[best_method]['mrr']:.4f})")

    # Save results to CSV
    output_path = Path("data/evaluation_results.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "hit_rate", "mrr", "total_queries"])
        writer.writeheader()
        for method, metrics in results.items():
            writer.writerow({"method": method, **metrics})

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
