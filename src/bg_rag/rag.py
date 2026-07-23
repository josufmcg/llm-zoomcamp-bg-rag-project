"""RAG pipeline for Baldur's Gate II character creation questions.

Implements the search → build context → build prompt → LLM call pipeline.
Adapted from the reference project's RAGBase pattern.

Usage:
    from bg_rag.rag import RAGPipeline
    from bg_rag.search import SearchEngine
    from bg_rag.embedder import Embedder
    from openai import OpenAI

    embedder = Embedder()
    search_engine = SearchEngine(embedder)
    client = OpenAI()
    rag = RAGPipeline(search_engine=search_engine, llm_client=client)
    answer = rag.ask("What class is best for a beginner?")
"""

from openai import OpenAI

from bg_rag.config import get_settings
from bg_rag.search import SearchEngine


INSTRUCTIONS = """
You are a knowledgeable guide for Baldur's Gate II: Shadows of Amn.
Your task is to answer questions about character creation, classes,
kits, abilities, and gameplay strategies.

Use ONLY the provided context to answer the question. The context
contains official class descriptions, kit details, and gameplay
recommendations from the game's FAQ.

If the answer is not found in the context, respond with:
"I don't have enough information to answer that question based on the available character data."

Keep your answers helpful, clear, and focused on the specific question asked.
Refer to specific class/kit names, abilities, and restrictions when relevant.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGPipeline:
    """RAG pipeline for answering BG2 character creation questions.

    Args:
        search_engine: Initialized SearchEngine instance.
        llm_client: OpenAI client instance.
        instructions: System instructions for the LLM.
        prompt_template: Template for building the user prompt.
        model: OpenAI model name.
        search_method: Default search method ("vector", "keyword", or "hybrid").
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        llm_client: OpenAI | None = None,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        model: str | None = None,
        search_method: str = "hybrid",
    ) -> None:
        self.search_engine = search_engine
        self.llm_client = llm_client or OpenAI()
        self.instructions = instructions
        self.prompt_template = prompt_template
        settings = get_settings()
        self.model = model or settings.llm_model
        self.search_method = search_method

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        """Search the knowledge base for relevant documents.

        Args:
            query: The user's question.
            num_results: Number of results to return.

        Returns:
            List of document dicts with scores.
        """
        return self.search_engine.search(
            query, method=self.search_method, limit=num_results
        )

    def build_context(self, search_results: list[dict]) -> str:
        """Build a text context from search results.

        Each result is formatted as:
            Category > Subcategory
            A: text

        Note: the documents table has no `question` column, so only the
        category, subcategory, and text fields are used in the context.

        Args:
            search_results: List of document dicts from search.

        Returns:
            Formatted context string.
        """
        lines = []
        for doc in search_results:
            lines.append(f"{doc['category']} > {doc['subcategory']}")
            lines.append(f"A: {doc['text']}")
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: list[dict]) -> str:
        """Build the full prompt from query and search results.

        Args:
            query: The user's question.
            search_results: List of document dicts from search.

        Returns:
            Formatted prompt string.
        """
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> str:
        """Call the OpenAI LLM and return the response text.

        Uses the Responses API (not Chat Completions).

        Args:
            prompt: The user prompt to send.

        Returns:
            The LLM's response text.
        """
        response = self.llm_client.responses.create(
            model=self.model,
            input=[
                {"role": "developer", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text

    def ask(self, query: str, search_method: str | None = None) -> str:
        """Run the full RAG pipeline: search → context → prompt → LLM.

        Args:
            query: The user's question.
            search_method: Override the default search method for this query.

        Returns:
            The LLM's answer text.
        """
        # Temporarily override search method if specified
        original_method = self.search_method
        if search_method:
            self.search_method = search_method

        try:
            search_results = self.search(query)
            prompt = self.build_prompt(query, search_results)
            answer = self.llm(prompt)
        finally:
            self.search_method = original_method

        return answer
