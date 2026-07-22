from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "bg_rag"
    postgres_user: str = "user"
    postgres_password: str = "password"

    # OpenAI
    openai_api_key: str = ""

    # Embedding model
    embedding_model_path: Path = Path("models/Xenova/all-MiniLM-L6-v2")

    # LLM
    llm_model: str = "gpt-4.1-mini"

    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"host={self.postgres_host} "
            f"port={self.postgres_port} "
            f"dbname={self.postgres_db} "
            f"user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


def get_settings() -> Settings:
    """Create and return a Settings instance."""
    return Settings()