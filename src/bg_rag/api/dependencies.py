"""FastAPI dependency injection.

Provides singleton instances of Embedder, SearchEngine, and RAGWithMetrics
that are initialized at application startup and shared across requests.
"""

from openai import OpenAI

from bg_rag.config import get_settings
from bg_rag.embedder import Embedder
from bg_rag.metrics import RAGWithMetrics
from bg_rag.search import SearchEngine

# Module-level singletons (initialized by init_dependencies)
_embedder: Embedder | None = None
_search_engine: SearchEngine | None = None
_rag: RAGWithMetrics | None = None


def init_dependencies() -> None:
    """Initialize all singleton dependencies.

    Called once at application startup (in the lifespan handler).
    """
    global _embedder, _search_engine, _rag

    settings = get_settings()

    _embedder = Embedder(settings.embedding_model_path)
    _search_engine = SearchEngine(_embedder)
    _rag = RAGWithMetrics(
        search_engine=_search_engine,
        llm_client=OpenAI(api_key=settings.openai_api_key),
        model=settings.llm_model,
    )


def get_embedder() -> Embedder:
    """Get the Embedder singleton."""
    assert _embedder is not None, "Dependencies not initialized. Call init_dependencies() first."
    return _embedder


def get_search_engine() -> SearchEngine:
    """Get the SearchEngine singleton."""
    assert _search_engine is not None, "Dependencies not initialized. Call init_dependencies() first."
    return _search_engine


def get_rag() -> RAGWithMetrics:
    """Get the RAGWithMetrics singleton."""
    assert _rag is not None, "Dependencies not initialized. Call init_dependencies() first."
    return _rag
