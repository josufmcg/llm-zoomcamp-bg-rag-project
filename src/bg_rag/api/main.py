"""FastAPI application entry point.

Creates the FastAPI app with lifespan management, CORS middleware,
and route registration.

Run with:
    uvicorn bg_rag.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bg_rag.api.dependencies import init_dependencies
from bg_rag.api.routes import documents, feedback, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Initializes dependencies (embedder, search engine, RAG pipeline)
    at startup.
    """
    print("Initializing dependencies...")
    init_dependencies()
    print("Dependencies initialized.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="BG RAG API",
    description="Baldur's Gate II RAG system for character creation questions",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware — allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(documents.router, tags=["Documents"])
app.include_router(rag.router, tags=["RAG"])
app.include_router(feedback.router, tags=["Feedback"])


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
