"""User feedback endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bg_rag.db import get_db_connection, now_utc


router = APIRouter()


class FeedbackRequest(BaseModel):
    """Request body for user feedback."""
    conversation_id: int
    score: int  # +1 (thumbs up) or -1 (thumbs down)


class FeedbackResponse(BaseModel):
    """Response from feedback submission."""
    status: str


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    """Submit user feedback (thumbs up/down) for a conversation.

    Args:
        request: Contains conversation_id and score (+1 or -1).

    Returns:
        Status confirmation.
    """
    if request.score not in (1, -1):
        raise HTTPException(
            status_code=400,
            detail="Score must be +1 (thumbs up) or -1 (thumbs down)",
        )

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Verify conversation exists
            cur.execute(
                "SELECT id FROM conversations WHERE id = %s",
                (request.conversation_id,),
            )
            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation {request.conversation_id} not found",
                )

            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, score, timestamp
                ) VALUES (%s, %s, %s, %s)
                """,
                (request.conversation_id, "user", request.score, now_utc()),
            )
        conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
    finally:
        conn.close()

    return FeedbackResponse(status="ok")
