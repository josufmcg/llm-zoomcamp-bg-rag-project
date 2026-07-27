"""RAG question-answering endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from bg_rag.api.dependencies import get_rag
from bg_rag.db import get_db_connection, now_utc
from bg_rag.judge import evaluate_relevance
from bg_rag.metrics import RAGWithMetrics


router = APIRouter()


class AskRequest(BaseModel):
    """Request body for RAG question-answering."""
    question: str
    search_method: str = "hybrid"


class AskResponse(BaseModel):
    """Response from RAG question-answering."""
    answer: str
    question: str
    model: str
    search_method: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    conversation_id: int
    relevance: str | None = None
    relevance_explanation: str | None = None


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    rag: RAGWithMetrics = Depends(get_rag),
):
    """Ask a question and get a RAG-generated answer.

    Runs the full pipeline:
    1. Search knowledge base (using specified method).
    2. Build context from search results.
    3. Generate answer via OpenAI.
    4. Save conversation to database.
    5. Run LLM-as-judge evaluation.
    6. Return answer with metrics.
    """
    # Run RAG pipeline
    try:
        answer = rag.ask(request.question, search_method=request.search_method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {str(e)}")

    record = rag.last_call
    if record is None:
        raise HTTPException(status_code=500, detail="No metrics recorded")

    # Save conversation to database
    timestamp = now_utc()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, model, search_method, prompt,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    request.question,
                    record.answer,
                    record.model,
                    request.search_method,
                    record.prompt,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()["id"]
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save conversation: {str(e)}")
    finally:
        conn.close()

    # Run LLM-as-judge evaluation (best effort — don't fail the request)
    relevance = None
    explanation = None
    try:
        relevance, explanation = evaluate_relevance(request.question, answer)

        # Save judge feedback
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback (
                        conversation_id, source, relevance, explanation, timestamp
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (conversation_id, "judge", relevance, explanation, now_utc()),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # Judge evaluation is best-effort; don't fail the main request
        pass

    return AskResponse(
        answer=answer,
        question=request.question,
        model=record.model,
        search_method=request.search_method,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        response_time=record.response_time,
        cost=record.cost,
        conversation_id=conversation_id,
        relevance=relevance,
        relevance_explanation=explanation,
    )
