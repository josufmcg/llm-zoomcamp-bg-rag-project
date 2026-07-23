"""Database connection management and schema initialization."""

import hashlib
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from bg_rag.config import get_settings

# Module-level connection pool (initialized lazily)
_pool = None


def get_db_connection() -> psycopg.Connection:
    """Get a new database connection using settings from config."""
    settings = get_settings()
    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        row_factory=dict_row,
    )


def init_db(drop: bool = False) -> None:
    """Initialize database tables and extensions.

    Args:
        drop: If True, drop existing tables before creating them.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            if drop:
                cur.execute("DROP TABLE IF EXISTS feedback CASCADE")
                cur.execute("DROP TABLE IF EXISTS conversations CASCADE")
                cur.execute("DROP TABLE IF EXISTS documents CASCADE")

            # Documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector(384),
                    text_search tsvector,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Update text_search column with weighted tsvector
            # This trigger automatically updates text_search when a row is inserted/updated
            cur.execute("""
                CREATE OR REPLACE FUNCTION documents_text_search_update() RETURNS trigger AS $$
                BEGIN
                    NEW.text_search :=
                        setweight(to_tsvector('english', COALESCE(NEW.subcategory, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(NEW.text, '')), 'B');
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql
            """)

            cur.execute("""
                DROP TRIGGER IF EXISTS documents_text_search_trigger ON documents
            """)

            cur.execute("""
                CREATE TRIGGER documents_text_search_trigger
                    BEFORE INSERT OR UPDATE ON documents
                    FOR EACH ROW
                    EXECUTE FUNCTION documents_text_search_update()
            """)

            # Indexes for documents
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_text_search
                ON documents USING GIN (text_search)
            """)

            # Conversations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    search_method TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time FLOAT NOT NULL,
                    cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)

            # Feedback table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    source TEXT NOT NULL,
                    relevance TEXT,
                    explanation TEXT,
                    score INTEGER,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)

        conn.commit()
        print("Database initialized successfully.")
    finally:
        conn.close()


def compute_doc_hash(category: str, subcategory: str, text: str) -> str:
    """Compute MD5 hash for document deduplication and doc_id generation.

    Args:
        category: Document category.
        subcategory: Document subcategory.
        text: Document text content.

    Returns:
        MD5 hex digest string.
    """
    content = f"{category}|{subcategory}|{text}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def now_utc() -> datetime:
    """Get current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print("Initializing database (drop and recreate)...")
    init_db(drop=True)
