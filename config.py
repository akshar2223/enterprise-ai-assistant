"""
Centralized application configuration using Pydantic Settings.
All config is loaded from environment variables / .env file.
Never hardcode secrets — always use these settings.
"""
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_chat_model: str = Field("gpt-4o", description="Chat completion model")
    openai_embedding_model: str = Field(
        "text-embedding-3-large", description="Embedding model"
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        ..., description="Full PostgreSQL connection URL"
    )
    postgres_host: str = Field("localhost")
    postgres_port: int = Field(5432)
    postgres_db: str = Field("enterprise_ai")
    postgres_user: str = Field("postgres")
    postgres_password: str = Field(...)

    # ── JWT Auth ──────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = Field("HS256")
    jwt_access_token_expire_minutes: int = Field(60)

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = Field("development")
    log_level: str = Field("INFO")
    api_rate_limit_per_minute: int = Field(60)

    # ── Agent Tuning ──────────────────────────────────────────────────────────
    vector_search_top_k: int = Field(5)
    faithfulness_threshold: float = Field(0.96)
    sql_query_max_rows: int = Field(100)

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Use this everywhere instead of instantiating Settings() directly.
    """
    return Settings()
