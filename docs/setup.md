# Setup Guide

Detailed instructions for setting up the BG2 RAG project for local development.

## Prerequisites

- **Python 3.12** — verify with `python --version`
- **uv** — Python package manager. Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker** — for PostgreSQL and Grafana. Install: https://docs.docker.com/get-docker/
- **OpenAI API key** — get one at https://platform.openai.com/api-keys

## Step 1: Clone and Setup

```bash
git clone <repo-url>
cd llm-zoomcamp-project

# Install Python dependencies
make setup
# This runs: uv sync + downloads the ONNX embedding model
```

## Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

The other defaults are fine for local development:
- `POSTGRES_HOST=localhost`
- `POSTGRES_DB=bg_rag`
- `POSTGRES_USER=user`
- `POSTGRES_PASSWORD=password`

## Step 3: Start PostgreSQL

In a dedicated terminal:

```bash
make db
```

This starts a `pgvector/pgvector:pg17` container on port 5432. Leave this terminal running.

## Step 4: Initialize Database

In another terminal:

```bash
make init-db
```

This creates the `documents`, `conversations`, and `feedback` tables with pgvector extension.

## Step 5: Generate Dataset (if not already done)

```bash
make dataset
```

This uses OpenAI to extract character class data from the GameFAQs BG2 FAQ and saves it to `data/bg_characters.json`.

## Step 6: Ingest Documents

```bash
make ingest
```

This loads the 11 character records, computes embeddings, and inserts them into PostgreSQL.

## Step 7: Start the Backend API

```bash
make dev-api
```

FastAPI starts on `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## Step 8: Start the Frontend

In another terminal:

```bash
make dev-streamlit
```

Streamlit starts on `http://localhost:8501`.

## Step 9: (Optional) Start the Dashboard

```bash
make dev-dashboard
```

Streamlit dashboard on `http://localhost:8502`.

## Step 10: (Optional) Run Evaluation

```bash
# Generate ground truth test questions
make ground-truth

# Evaluate search quality
make evaluate
```

## Docker Setup (Alternative)

If you prefer running everything in Docker:

```bash
make build          # Build images
make up             # Start all services
make docker-init-db # Initialize database
make docker-ingest  # Ingest documents
```

Services:
- API: http://localhost:8000
- Streamlit: http://localhost:8501
- Grafana: http://localhost:3000 (admin/admin)

## Troubleshooting

### "Connection refused" when connecting to PostgreSQL

Make sure the PostgreSQL container is running:
```bash
docker ps | grep bg-rag-postgres
```

### "OPENAI_API_KEY not set"

Check your `.env` file has `OPENAI_API_KEY=sk-...` and that you're running from the project root.

### "tokenizer.json not found"

Run the model download:
```bash
uv run python scripts/download_model.py
```

### Port already in use

Kill the existing process:
```bash
lsof -i :8000  # or :8501, :5432
kill <PID>
```
