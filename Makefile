.PHONY: setup db init-db ingest dev-api dev-streamlit dev-dashboard build up down logs ground-truth evaluate sample-data

# ── Setup ──────────────────────────────────────────
setup:
	uv sync
	uv run python scripts/download_model.py

# ── Local Development ──────────────────────────────
db:
	docker run -it --rm \
		--name bg-rag-postgres \
		-e POSTGRES_USER=user \
		-e POSTGRES_PASSWORD=password \
		-e POSTGRES_DB=bg_rag \
		-v bg_rag_pgdata:/var/lib/postgresql/data \
		-p 5432:5432 \
		pgvector/pgvector:pg17

init-db:
	uv run python -c "from bg_rag.db import init_db; init_db()"

ingest:
	uv run python -m bg_rag.ingest

dev-api:
	uv run uvicorn bg_rag.api.main:app --reload --host 0.0.0.0 --port 8000

dev-streamlit:
	cd src && uv run streamlit run bg_rag/frontend/app.py --server.port 8501

dev-dashboard:
	cd src && uv run streamlit run bg_rag/frontend/dashboard.py --server.port 8502

# ── Docker ─────────────────────────────────────────
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# ── Evaluation ─────────────────────────────────────
ground-truth:
	uv run python scripts/generate_ground_truth.py

evaluate:
	uv run python scripts/evaluate.py

sample-data:
	uv run python scripts/generate_sample_data.py
