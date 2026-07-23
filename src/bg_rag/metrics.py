"""RAG pipeline with metrics tracking.

Extends RAGPipeline to record token usage, response time, and cost
for each LLM call.

Usage:
    from bg_rag.metrics import RAGWithMetrics
    rag = RAGWithMetrics(search_engine=engine, llm_client=client)
    answer = rag.ask("What is the best fighter kit?")
    record = rag.last_call  # LLMCallRecord with metrics
"""

import time

from openai import OpenAI

from bg_rag.models import LLMCallRecord
from bg_rag.rag import RAGPipeline
from bg_rag.search import SearchEngine


# gpt-4.1-mini pricing (per million tokens)
PRICING = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM API call.

    Args:
        model: The model name (e.g., "gpt-4.1-mini").
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = PRICING.get(model, PRICING["gpt-4.1-mini"])
    cost = (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000
    return cost


class RAGWithMetrics(RAGPipeline):
    """RAG pipeline that tracks metrics for each LLM call.

    After calling ask(), the metrics are available in self.last_call
    as an LLMCallRecord dataclass.

    Args:
        search_engine: Initialized SearchEngine instance.
        llm_client: OpenAI client instance.
        **kwargs: Additional arguments passed to RAGPipeline.
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        llm_client: OpenAI | None = None,
        **kwargs,
    ) -> None:
        super().__init__(search_engine=search_engine, llm_client=llm_client, **kwargs)
        self.last_call: LLMCallRecord | None = None

    def llm(self, prompt: str) -> str:
        """Call the LLM and record metrics.

        Overrides RAGPipeline.llm() to add timing and token tracking.

        Args:
            prompt: The user prompt to send.

        Returns:
            The LLM's response text.
        """
        start_time = time.time()

        response = self.llm_client.responses.create(
            model=self.model,
            input=[
                {"role": "developer", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
        )

        response_time = time.time() - start_time
        usage = response.usage
        cost = calculate_cost(
            self.model, usage.input_tokens, usage.output_tokens
        )

        self.last_call = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            answer=response.output_text,
            search_method=self.search_method,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_time=response_time,
            cost=cost,
        )

        return response.output_text
