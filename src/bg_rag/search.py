"""Search engine with vector, keyword, and hybrid search methods.

All search methods query the PostgreSQL documents table and return
results as a list of dicts with document fields plus a relevance score.

Usage:
    from bg_rag.search import SearchEngine
    from bg_rag.embedder import Embedder

    embedder = Embedder()
    engine = SearchEngine(embedder)
    results = engine.hybrid_search("What class should I pick for melee combat?")
"""

from bg_rag.db import get_db_connection
from bg_rag.embedder import Embedder


class SearchEngine:
    """Search engine supporting vector, keyword, and hybrid search.

    Args:
        embedder: Initialized Embedder instance for query vectorization.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def vector_search(self, query: str, limit: int = 5) -> list[dict]:
        """Search using cosine similarity on vector embeddings.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.

        Returns:
            List of document dicts with an added 'score' field (0-1, higher is better).
        """
        query_embedding = self.embedder.encode(query)

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id, doc_id, category, subcategory, text,
                        1 - (embedding <=> %s::vector) AS score
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query_embedding.tolist(), limit),
                )
                results = cur.fetchall()
        finally:
            conn.close()

        return [dict(row) for row in results]

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        """Search using PostgreSQL full-text search with ranking.

        Uses the pre-computed text_search tsvector column with
        weighted fields (subcategory=A, text=B).

        Args:
            query: The search query text.
            limit: Maximum number of results to return.

        Returns:
            List of document dicts with an added 'score' field (higher is better).
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id, doc_id, category, subcategory, text,
                        ts_rank(text_search, plainto_tsquery('english', %s)) AS score
                    FROM documents
                    WHERE text_search @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, query, limit),
                )
                results = cur.fetchall()
        finally:
            conn.close()

        return [dict(row) for row in results]

    def hybrid_search(self, query: str, limit: int = 5, rrf_k: int = 60) -> list[dict]:
        """Search using Reciprocal Rank Fusion of vector + keyword search.

        Runs both vector and keyword search, then combines results using
        RRF: score(doc) = sum(1 / (k + rank_i)) for each ranking the
        document appears in.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.
            rrf_k: The RRF constant (default 60, standard value).

        Returns:
            List of document dicts with an added 'score' field (RRF score).
        """
        # Get more results from each method to ensure good fusion
        fetch_limit = limit * 3

        vector_results = self.vector_search(query, limit=fetch_limit)
        keyword_results = self.keyword_search(query, limit=fetch_limit)

        # Build RRF scores
        # Key: doc_id (the database primary key), Value: rrf_score
        rrf_scores: dict[int, float] = {}
        doc_data: dict[int, dict] = {}

        for rank, doc in enumerate(vector_results):
            doc_pk = doc["id"]
            rrf_scores[doc_pk] = rrf_scores.get(doc_pk, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_data[doc_pk] = doc

        for rank, doc in enumerate(keyword_results):
            doc_pk = doc["id"]
            rrf_scores[doc_pk] = rrf_scores.get(doc_pk, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_data[doc_pk] = doc

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda pk: rrf_scores[pk], reverse=True)

        results = []
        for pk in sorted_ids[:limit]:
            doc = doc_data[pk].copy()
            doc["score"] = rrf_scores[pk]
            results.append(doc)

        return results

    def search(self, query: str, method: str = "hybrid", limit: int = 5) -> list[dict]:
        """Dispatch search to the specified method.

        Args:
            query: The search query text.
            method: One of "vector", "keyword", or "hybrid".
            limit: Maximum number of results to return.

        Returns:
            List of document dicts with scores.

        Raises:
            ValueError: If method is not recognized.
        """
        if method == "vector":
            return self.vector_search(query, limit=limit)
        elif method == "keyword":
            return self.keyword_search(query, limit=limit)
        elif method == "hybrid":
            return self.hybrid_search(query, limit=limit)
        else:
            raise ValueError(
                f"Unknown search method: {method}. "
                "Use 'vector', 'keyword', or 'hybrid'."
            )
