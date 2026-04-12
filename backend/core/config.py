from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── LLM API Keys ─────────────────────────────────────
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    openrouter_api_key: str = ""

    # ── LLM Routing ──────────────────────────────────────
    llm_code_model: str = "deepseek-coder"
    llm_reasoning_model: str = "gemini-2.0-flash"
    llm_security_model: str = "deepseek-reasoner"
    llm_embedding_model: str = "models/gemini-embedding-001"
    embedding_vector_dim: int = 768
    llm_fallback_model: str = "openrouter"

    # ── PostgreSQL ────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "codebase_intelligence"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # ── Redis ─────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Qdrant ───────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ── Neo4j ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def resolved_embedding_model(self) -> str:
        """
        Normalize legacy Gemini embedding model names to currently supported ones.
        """
        legacy_model_map = {
            "models/text-embedding-004": "models/gemini-embedding-001",
        }
        return legacy_model_map.get(self.llm_embedding_model, self.llm_embedding_model)

    def validate_api_keys(self) -> list[str]:
        """
        Returns a list of warning messages for missing API keys.
        Called at startup — missing keys disable that provider gracefully.
        """
        warnings = []
        if not self.gemini_api_key:
            warnings.append("GEMINI_API_KEY not set — embeddings and primary LLM will be unavailable.")
        if not self.deepseek_api_key:
            warnings.append("DEEPSEEK_API_KEY not set — code-specific reasoning will use Gemini fallback.")
        if not self.openrouter_api_key:
            warnings.append("OPENROUTER_API_KEY not set — fallback provider disabled.")
        return warnings

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()