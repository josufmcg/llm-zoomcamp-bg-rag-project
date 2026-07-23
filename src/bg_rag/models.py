"""Shared data models and dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call with metrics.

    Used to track conversation metrics for storage in the
    conversations table and display in the UI.
    """

    model: str
    prompt: str
    answer: str
    search_method: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
