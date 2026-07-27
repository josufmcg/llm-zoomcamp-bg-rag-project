.PHONY: setup db init-db ingest dev-api dev-streamlit dev-dashboard build up down logs \
        ground-truth evaluate sample-data clean help

# ── Help ───────────────────────────────────────────
help:
	@echo "BG RAG Project — Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup           Install deps + download ONNX model"
	@echo "    make db              Start local PostgreSQL container"
	@echo "    make init-db         Create database tables"
	@echo "    make ingest          Ingest bg_characters.json into DB"
	@echo ""
	@echo "  Development:"
	@echo "    make dev-api         Run FastAPI on port 8000"
	@echo "    make dev-streamlit   Run Streamlit on port 8501"
	@echo "    make dev-dashboard   Run Dashboard on port 8502"
	@echo ""
	@echo "  Docker:"
	@echo "    make build           Build all Docker images"
	@echo "    make up              Start all services"
	@echo "    make down            Stop all services"
	@echo "    make logs            Tail logs from all services"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make ground-truth    Generate ground truth Q&A pairs"
	@echo "    make evaluate        Run search evaluation metrics"
	@echo "    make sample-data     Generate fake data for dashboard"
	@echo ""
	@echo "  Other:"
	@echo "    make clean           Remove containers and volumes"

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
	uv run python -c "from bg_rag.db import init_db; init_db(drop=True)"

ingest:
	uv run python -c "from bg_rag.ingest import main; main()"

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

# ── Docker init (run after 'make up') ──────────────
docker-init-db:
	docker compose exec api python -c "from bg_rag.db import init_db; init_db(drop=True)"

docker-ingest:
	docker compose exec api python -c "from bg_rag.ingest import main; main()"

# ── Evaluation ─────────────────────────────────────
ground-truth:
	uv run python scripts/generate_ground_truth.py

evaluate:
	uv run python scripts/evaluate.py

sample-data:
	uv run python scripts/generate_sample_data.py

# ── Dataset ────────────────────────────────────────
dataset:
	uv run python scripts/generate_dataset.py

# ── Cleanup ────────────────────────────────────────
clean:
	docker compose down -v
	docker volume rm bg_rag_pgdata 2>/dev/null || true
