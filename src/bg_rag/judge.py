"""LLM-as-judge for relevance evaluation.

Uses OpenAI structured output to classify the relevance of a
RAG-generated answer to the original question.

Adapted from the reference project's judge.py pattern.
"""

import time

from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

from bg_rag.config import get_settings


class RelevanceVerdict(BaseModel):
    """Structured output for relevance judgment."""
    relevance: Literal["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
    explanation: str


JUDGE_INSTRUCTIONS = """
You are an expert evaluator for a RAG system about Baldur's Gate II character creation.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: The answer directly and accurately addresses the question using correct BG2 information.
- PARTLY_RELEVANT: The answer partially addresses the question or includes some relevant information.
- NON_RELEVANT: The answer does not address the question or provides incorrect information.
""".strip()

JUDGE_PROMPT = """
Question: {question}
Generated Answer: {answer}
""".strip()


def evaluate_relevance(
    question: str,
    answer: str,
    client: OpenAI | None = None,
    max_retries: int = 3,
) -> tuple[str, str]:
    """Evaluate the relevance of a RAG answer to a question.

    Args:
        question: The original user question.
        answer: The RAG-generated answer.
        client: OpenAI client (created from settings if None).
        max_retries: Number of retry attempts on failure.

    Returns:
        Tuple of (relevance, explanation) where relevance is one of
        "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT".
    """
    if client is None:
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

    prompt = JUDGE_PROMPT.format(question=question, answer=answer)

    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model="gpt-4.1-mini",
                input=[
                    {"role": "developer", "content": JUDGE_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                text_format=RelevanceVerdict,
            )
            result = response.output_parsed
            return result.relevance, result.explanation
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    # Should not reach here, but just in case
    return "NON_RELEVANT", "Evaluation failed"
