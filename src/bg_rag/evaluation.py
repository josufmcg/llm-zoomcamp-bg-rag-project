"""Evaluation metrics for search quality.

Implements Hit Rate and Mean Reciprocal Rank (MRR) for evaluating
search result relevance against ground truth data.
"""


def hit_rate_at_k(search_results: list[dict], target_doc_id: str, k: int = 5) -> float:
    """Compute Hit Rate@K for a single query.

    Hit Rate@K = 1 if the target document appears in the top K results, else 0.

    Args:
        search_results: List of search result dicts (must have 'doc_id' field).
        target_doc_id: The expected document ID.
        k: Number of top results to consider.

    Returns:
        1.0 if hit, 0.0 if miss.
    """
    top_k_ids = [r["doc_id"] for r in search_results[:k]]
    return 1.0 if target_doc_id in top_k_ids else 0.0


def mrr_at_k(search_results: list[dict], target_doc_id: str, k: int = 5) -> float:
    """Compute Mean Reciprocal Rank (MRR@K) for a single query.

    MRR@K = 1/rank if the target document is in top K results, else 0.

    Args:
        search_results: List of search result dicts (must have 'doc_id' field).
        target_doc_id: The expected document ID.
        k: Number of top results to consider.

    Returns:
        Reciprocal rank (1/rank) or 0.0 if not found in top K.
    """
    for i, result in enumerate(search_results[:k]):
        if result["doc_id"] == target_doc_id:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_search(
    ground_truth: list[dict],
    search_func,
    k: int = 5,
) -> dict[str, float]:
    """Evaluate a search function against ground truth data.

    Args:
        ground_truth: List of dicts with 'doc_id' and 'question' keys.
        search_func: A callable that takes a query string and returns
                      a list of result dicts with 'doc_id' field.
        k: Number of top results to consider.

    Returns:
        Dict with 'hit_rate' and 'mrr' average scores.
    """
    hit_rates = []
    mrrs = []

    for item in ground_truth:
        results = search_func(item["question"])
        hr = hit_rate_at_k(results, item["doc_id"], k=k)
        mrr = mrr_at_k(results, item["doc_id"], k=k)
        hit_rates.append(hr)
        mrrs.append(mrr)

    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

    return {
        "hit_rate": avg_hit_rate,
        "mrr": avg_mrr,
        "total_queries": len(ground_truth),
    }
