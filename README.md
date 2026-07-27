# ⚔️ Baldur's Gate II — RAG Character Guide

A Retrieval-Augmented Generation (RAG) system that answers questions about Baldur's Gate II: Shadows of Amn character creation. Built as a learning project for the [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course.

## 🎯 What Does It Do?

Ask questions about BG2 character classes (Fighter, Mage, Thief, etc.), their kits, abilities, and strategies — and get AI-generated answers grounded in actual game FAQ data.

**Example:**
> **Q:** What is the best Fighter kit for dealing maximum damage?
>
> **A:** The Kensai is considered the top Fighter kit for damage output. With +1 to hit and damage for every three levels, a -2 AC bonus, and the Kai ability that makes all attacks deal maximum damage for 10 seconds, the Kensai excels at raw damage. However, they cannot wear armor or use missile weapons, so pairing with 18 DEX is recommended...

## 🏗️ Architecture

```
┌─────────────┐     HTTP      ┌──────────────┐     SQL/pgvector    ┌──────────────┐
│  Streamlit   │◄────────────►│   FastAPI     │◄──────────────────►│  PostgreSQL   │
│  (Frontend)  │              │   (Backend)   │                    │  + pgvector   │
│  port 8501   │              │   port 8000   │                    │  port 5432    │
└─────────────┘              └──────────────┘                    └──────────────┘
                                    │                                    ▲
                                    │ OpenAI API                         │
                                    ▼                              ┌─────┴────────┐
                              ┌──────────┐                        │   Grafana     │
                              │  OpenAI  │                        │   port 3000   │
                              └──────────┘                        └──────────────┘
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI | REST API for ingestion, search, RAG, feedback |
| Frontend | Streamlit | Conversation UI + metrics dashboard |
| Database | PostgreSQL + pgvector | Document storage, vector search, conversation tracking |
| Embeddings | ONNX (all-MiniLM-L6-v2) | Local 384-dim text embeddings |
| LLM | OpenAI gpt-4.1-mini | Answer generation + evaluation |
| Monitoring | Grafana | Auto-provisioned metrics dashboard |

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Run with Docker (recommended)

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd llm-zoomcamp-project

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 3. Build and start all services
make build
make up

# 4. Initialize database and ingest data
make docker-init-db
make docker-ingest

# 5. Open the app
# Streamlit UI: http://localhost:8501
# API docs:     http://localhost:8000/docs
# Grafana:      http://localhost:3000 (admin/admin)
```

### Run locally (development)

See [docs/setup.md](docs/setup.md) for detailed local development instructions.

## 📊 Features

- **Three search methods:** Vector (semantic), Keyword (full-text), Hybrid (RRF fusion)
- **RAG with metrics:** Tracks tokens, cost, response time for every conversation
- **LLM-as-judge:** Automatic relevance evaluation (RELEVANT / PARTLY_RELEVANT / NON_RELEVANT)
- **User feedback:** Thumbs up/down on every answer
- **Grafana dashboard:** Auto-provisioned with cost, performance, and quality metrics
- **Search evaluation:** Hit Rate and MRR metrics with ground truth Q&A pairs

## 🗂️ Dataset

11 character class records from the [BG2 Character FAQ](https://gamefaqs.gamespot.com/pc/663934-baldurs-gate-ii-enhanced-edition/faqs/14105):

Fighter, Ranger, Paladin, Barbarian, Cleric, Druid, Monk, Thief, Bard, Mage, Sorcerer

Each record includes: class description, abilities, restrictions, kit/subclass variants, and gameplay recommendations.

## 📁 Project Structure

```
src/bg_rag/           # Main Python package
├── config.py         # Configuration (env vars)
├── db.py             # Database connection & schema
├── embedder.py       # ONNX text embedding
├── search.py         # Vector, keyword, hybrid search
├── rag.py            # RAG pipeline
├── metrics.py        # Metrics tracking
├── judge.py          # LLM-as-judge evaluation
├── api/              # FastAPI backend
└── frontend/         # Streamlit UI + dashboard
```

## 🧪 Evaluation

```bash
# Generate ground truth questions (requires OpenAI API)
make ground-truth

# Run search evaluation
make evaluate
```

Outputs Hit Rate@5 and MRR@5 for all three search methods.

## 📖 Documentation

- [Setup Guide](docs/setup.md) — Detailed local development setup
- [Usage Guide](docs/usage.md) — How to use all features
- [Project Plan](plan/000.bg%20rag%20general%20plan.md) — Full technical plan

## 📝 License

This project is for educational purposes (LLM Zoomcamp course project).
