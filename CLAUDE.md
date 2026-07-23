# CLAUDE.md — AI Coding Guidelines for BG RAG Project

This file provides instructions for AI coding assistants (Copilot, Claude, etc.) working on this project. Read this file before making any changes.

## Project Overview

Baldur's Gate II RAG system — a Python application that answers questions about BG2 character creation using Retrieval-Augmented Generation. See `plan/000.bg rag general plan.md` for the full plan.

## Architecture

- **FastAPI backend** (port 8000): REST API for document ingestion, search, RAG Q&A, feedback.
- **Streamlit frontend** (port 8501): Thin UI client that calls FastAPI via HTTP.
- **PostgreSQL + pgvector** (port 5432): Single database for documents (with embeddings), conversations, feedback.
- **Grafana** (port 3000): Auto-provisioned monitoring dashboard reading from PostgreSQL.

## Tech Stack

- **Python 3.12** with **uv** package manager
- **FastAPI** + **uvicorn** for the backend API
- **Streamlit** for the frontend
- **PostgreSQL 17** with **pgvector** extension for vector search
- **ONNX Runtime** + **Tokenizers** for local embeddings (`Xenova/all-MiniLM-L6-v2`, 384 dimensions)
- **OpenAI API** (`gpt-4.1-mini`) for LLM generation and evaluation
- **psycopg** (v3) for database access (with connection pooling)
- **pydantic** / **pydantic-settings** for models and configuration
- **Docker** + **docker-compose** for containerization

## Project Layout

```
src/bg_rag/                    # Main Python package
├── config.py                  # pydantic-settings: env vars, constants
├── db.py                      # DB connection pool, init, migrations
├── embedder.py                # ONNX embedding model wrapper
├── models.py                  # Pydantic models & dataclasses
├── ingest.py                  # Document ingestion logic
├── search.py                  # Vector, keyword, hybrid search
├── rag.py                     # RAG pipeline (search → context → LLM)
├── metrics.py                 # RAGWithMetrics (token/cost tracking)
├── judge.py                   # LLM-as-judge relevance evaluation
├── evaluation.py              # Ground truth gen, Hit Rate, MRR
├── api/                       # FastAPI application
│   ├── main.py                # App creation & lifespan
│   ├── dependencies.py        # Dependency injection
│   └── routes/
│       ├── documents.py       # POST /documents, GET /search
│       ├── rag.py             # POST /ask
│       └── feedback.py        # POST /feedback
└── frontend/                  # Streamlit application
    ├── app.py                 # Conversation UI
    └── dashboard.py           # Metrics dashboard
```

## Coding Conventions

### Style & Formatting
- Use **type hints** on all function signatures and class attributes.
- Use **f-strings** for string formatting (never `.format()` or `%`).
- Use **pathlib.Path** instead of `os.path` for file path operations.
- Use **dataclasses** for simple data containers, **Pydantic models** for API request/response schemas and configuration.
- Imports order: stdlib → third-party → local (separated by blank lines).
- No wildcard imports (`from x import *`).
- Maximum line length: 100 characters.
- Use double quotes for strings.

### Python Patterns
- **Configuration:** All configuration via environment variables, loaded through `pydantic-settings` in `src/bg_rag/config.py`. Never hardcode connection strings, API keys, or model paths.
- **Database access:** Always use `src/bg_rag/db.py` functions. Use `psycopg` (v3, not v2). Use connection pooling via `psycopg_pool.ConnectionPool`. Always use parameterized queries (`%s` placeholders, never f-strings in SQL).
- **Embeddings:** Use `src/bg_rag/embedder.py` (ONNX-based). Initialize once at startup, reuse the instance. Never import sentence-transformers or call OpenAI for embeddings.
- **OpenAI calls:** Use the **Responses API** (`client.responses.create()` and `client.responses.parse()`), NOT the Chat Completions API. Use `role: "developer"` for system instructions (not `role: "system"`). Use `input=` parameter (not `messages=`).
- **Error handling:** Use specific exceptions, not bare `except:`. Log errors with context. In FastAPI routes, raise `HTTPException` with appropriate status codes.
- **Async:** FastAPI routes are **synchronous** (def, not async def) since we use psycopg in sync mode. The embedder and OpenAI calls are also synchronous.

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Database tables: `snake_case` (plural: `documents`, `conversations`, `feedback`)
- API endpoints: `lowercase` with no trailing slash

### Database Conventions
- Single PostgreSQL database named `bg_rag`.
- pgvector extension for vector columns (`vector(384)` type).
- MD5 hash (`doc_id` column) for document deduplication — computed from `category + subcategory + text`.
- Use `SERIAL PRIMARY KEY` for auto-increment IDs.
- Use `TIMESTAMP WITH TIME ZONE` for all timestamp columns.
- Full-text search via `tsvector` column with GIN index.

### Search Implementation
- **Vector search:** Use pgvector's `<=>` operator (cosine distance). Compute `1 - distance` for similarity score.
- **Keyword search:** Use `to_tsvector('english', ...)` and `plainto_tsquery('english', ...)` with `ts_rank()`. Weight fields: subcategory=A, text=B.
- **Hybrid search:** Reciprocal Rank Fusion (RRF) with k=60: `score = Σ 1/(60 + rank)`. Normalize to 0-1 range.

### RAG Pipeline
- Prompt template includes: system instructions (BG2 domain-specific), user question, and search result context.
- Context is built from search results: each result formatted as "Category > Subcategory\nQ: question\nA: text\n".
- Cost calculation for `gpt-4.1-mini`: input $0.40/M tokens, output $1.60/M tokens.
- Track all metrics in `LLMCallRecord` dataclass.

### Docker
- Base image: `python:3.12-slim`.
- Install uv from `ghcr.io/astral-sh/uv:latest` via COPY.
- Copy `pyproject.toml` and `uv.lock` first, then `RUN uv sync --locked` (layer caching).
- Set `ENV PATH="/app/.venv/bin:$PATH"`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `bg_rag` | Database name |
| `POSTGRES_USER` | `user` | Database user |
| `POSTGRES_PASSWORD` | `password` | Database password |
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `EMBEDDING_MODEL_PATH` | `models/Xenova/all-MiniLM-L6-v2` | Path to ONNX embedding model |
| `LLM_MODEL` | `gpt-4.1-mini` | OpenAI model name |

## Common Commands

```bash
# Setup
uv sync                           # Install dependencies
uv run python scripts/download_model.py  # Download ONNX model

# Local development (needs PostgreSQL running)
make db                           # Start PostgreSQL container
make init-db                      # Create database tables
make ingest                       # Ingest bg_characters.json
make dev-api                      # Run FastAPI on port 8000
make dev-streamlit                # Run Streamlit on port 8501

# Docker
make build                        # Build all Docker images
make up                           # Start all services
make down                         # Stop all services
make logs                         # Tail logs

# Evaluation
make ground-truth                 # Generate ground truth Q&A pairs
make evaluate                     # Run search evaluation metrics
```

## Task Files

Implementation is split into 13 tasks in `plan/`. Execute them **in order** — each task lists its dependencies. Each task file contains:
1. **Objective** — what this task produces
2. **Dependencies** — which prior tasks must be complete
3. **Files to create/modify** — exact file paths
4. **Detailed instructions** — step-by-step implementation guide with code snippets
5. **Verification checklist** — specific commands to confirm the task is done correctly

## Important Reminders

- **Never commit `.env` files** or ONNX model binaries to git.
- **Always read the relevant task file** before implementing. The task files contain exact schemas, code patterns, and verification steps.
- **Follow the reference project patterns** from `llm-zoomcamp/05.monitoring/` — this project is an adaptation of those patterns for BG2 data with FastAPI instead of Streamlit-only.
- **Use the OpenAI Responses API**, not the legacy Chat Completions API. The reference code uses `client.responses.create()` and `client.responses.parse()`.
- **The embedding dimension is 384** (all-MiniLM-L6-v2). Never change this without updating the database schema.
