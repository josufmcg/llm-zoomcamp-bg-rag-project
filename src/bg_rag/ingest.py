"""Document ingestion pipeline.

Loads documents from a JSON file, computes embeddings and MD5 hashes,
and inserts them into the PostgreSQL documents table with deduplication.

The documents table stores doc_id (MD5 hash of category|subcategory|text),
category, subcategory, text, and a 384-dim embedding. The question field
from the dataset is used only to enrich the embedding text, not stored.

Usage:
    # As a module:
    uv run python -m bg_rag.ingest

    # Or via Makefile:
    make ingest
"""

import json
from pathlib import Path

from bg_rag.config import get_settings
from bg_rag.db import compute_doc_hash, get_db_connection
from bg_rag.embedder import Embedder


def load_documents(path: str | Path = "data/bg_characters.json") -> list[dict]:
    """Load documents from a JSON file.

    Args:
        path: Path to the JSON file containing document records.

    Returns:
        List of document dicts with keys: id, category, subcategory, question, text.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run 'uv run python scripts/generate_dataset.py' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_documents(
    documents: list[dict],
    embedder: Embedder,
) -> tuple[int, int]:
    """Ingest documents into PostgreSQL with embeddings and deduplication.

    For each document:
    1. Compute MD5 hash from category + subcategory + text (stored as doc_id).
    2. Skip if a document with the same doc_id already exists.
    3. Compute embedding vector from concatenation of question + text
       (the question enriches the embedding semantics but is not stored).
    4. Insert into the documents table.

    Args:
        documents: List of document dicts.
        embedder: Initialized Embedder instance.

    Returns:
        Tuple of (ingested_count, skipped_count).
    """
    conn = get_db_connection()
    ingested = 0
    skipped = 0

    try:
        with conn.cursor() as cur:
            for doc in documents:
                doc_hash = compute_doc_hash(
                    doc["category"],
                    doc["subcategory"],
                    doc["text"],
                )

                # Check for existing document with same doc_id (the hash)
                cur.execute(
                    "SELECT id FROM documents WHERE doc_id = %s",
                    (doc_hash,),
                )
                if cur.fetchone() is not None:
                    print(f"  Skipping {doc['id']} (already exists)")
                    skipped += 1
                    continue

                # Compute embedding from question + text combined.
                # The question adds semantic signal even though it is not stored.
                embedding_text = f"{doc['question']} {doc['text']}"
                embedding = embedder.encode(embedding_text)

                # Insert document (no question column in the schema)
                cur.execute(
                    """
                    INSERT INTO documents (
                        doc_id, category, subcategory, text, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s::vector
                    )
                    """,
                    (
                        doc_hash,
                        doc["category"],
                        doc["subcategory"],
                        doc["text"],
                        embedding.tolist(),
                    ),
                )
                ingested += 1
                print(f"  Ingested {doc['id']}")

        conn.commit()
    finally:
        conn.close()

    return ingested, skipped


def create_vector_index() -> None:
    """Create the pgvector HNSW index on the documents table.

    This should be called after bulk ingestion is complete.
    HNSW index provides approximate nearest neighbor search
    with good recall and fast queries.

    Note: For 11 documents, exact search is fine. The index
    is created for consistency and to demonstrate the pattern
    for larger datasets.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding
                ON documents USING hnsw (embedding vector_cosine_ops)
            """)
        conn.commit()
        print("  Vector index created.")
    finally:
        conn.close()


def main() -> None:
    """Main ingestion pipeline entry point."""
    settings = get_settings()

    print("Loading documents...")
    documents = load_documents()
    print(f"  Loaded {len(documents)} documents")

    print("Initializing embedder...")
    embedder = Embedder(settings.embedding_model_path)

    print("Ingesting documents...")
    ingested, skipped = ingest_documents(documents, embedder)
    print(f"  Done: {ingested} ingested, {skipped} skipped")

    if ingested > 0:
        print("Creating vector index...")
        create_vector_index()

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
