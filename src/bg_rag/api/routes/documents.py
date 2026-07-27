"""Document ingestion and search endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from bg_rag.api.dependencies import get_embedder, get_search_engine
from bg_rag.db import compute_doc_hash, get_db_connection
from bg_rag.embedder import Embedder
from bg_rag.search import SearchEngine


router = APIRouter()


# ── Request/Response models ───────────────────────────

class DocumentInput(BaseModel):
    """A single document to ingest."""
    category: str
    subcategory: str
    question: str  # Used to enrich the embedding; NOT stored in the table
    text: str


class IngestRequest(BaseModel):
    """Request body for document ingestion."""
    documents: list[DocumentInput]


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    ingested: int
    skipped: int


class SearchResult(BaseModel):
    """A single search result."""
    id: int
    doc_id: str
    category: str
    subcategory: str
    text: str
    score: float


class SearchResponse(BaseModel):
    """Response from search."""
    results: list[SearchResult]
    method: str
    query: str


# ── Endpoints ─────────────────────────────────────────

@router.post("/documents", response_model=IngestResponse)
def ingest_documents(
    request: IngestRequest,
    embedder: Embedder = Depends(get_embedder),
):
    """Ingest one or more documents into the knowledge base.

    Each document is:
    1. Hashed (MD5 of category + subcategory + text) for deduplication.
       The hash is stored as `doc_id`.
    2. Skipped if a document with the same doc_id already exists.
    3. Embedded (question + text combined) using the ONNX model. The
       `question` field enriches the embedding but is NOT stored.
    4. Inserted into the documents table.
    """
    conn = get_db_connection()
    ingested = 0
    skipped = 0

    try:
        with conn.cursor() as cur:
            for doc in request.documents:
                doc_hash = compute_doc_hash(
                    doc.category, doc.subcategory, doc.text
                )

                # Check for existing document (doc_id is the hash)
                cur.execute(
                    "SELECT id FROM documents WHERE doc_id = %s",
                    (doc_hash,),
                )
                if cur.fetchone() is not None:
                    skipped += 1
                    continue

                # Compute embedding from question + text combined
                embedding_text = f"{doc.question} {doc.text}"
                embedding = embedder.encode(embedding_text)

                # Insert (no question column in the schema)
                cur.execute(
                    """
                    INSERT INTO documents (
                        doc_id, category, subcategory, text, embedding
                    ) VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (
                        doc_hash,
                        doc.category,
                        doc.subcategory,
                        doc.text,
                        embedding.tolist(),
                    ),
                )
                ingested += 1

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        conn.close()

    return IngestResponse(ingested=ingested, skipped=skipped)


@router.get("/search", response_model=SearchResponse)
def search_documents(
    q: str = Query(..., description="Search query text"),
    method: str = Query("hybrid", description="Search method: vector, keyword, or hybrid"),
    limit: int = Query(5, description="Maximum number of results", ge=1, le=20),
    search_engine: SearchEngine = Depends(get_search_engine),
):
    """Search the knowledge base using vector, keyword, or hybrid search."""
    try:
        results = search_engine.search(q, method=method, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        method=method,
        query=q,
    )
