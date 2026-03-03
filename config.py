"""Centralized configuration for Qdrant Proxy.

All environment variables are loaded here and exposed via a Settings instance.
This replaces scattered os.getenv() calls throughout the codebase.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Qdrant Configuration
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    collection_name: str = Field(
        default="three-stage-search", alias="COLLECTION_NAME"
    )

    # Ingestion Filtering
    min_content_words: int = Field(
        default=32,
        alias="MIN_CONTENT_WORDS",
        description="Minimum word count required to store a document in Qdrant",
    )

    # Dense Embedding Configuration
    dense_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B", alias="DENSE_MODEL_NAME"
    )
    dense_vector_size: int = Field(
        default=1024, alias="DENSE_VECTOR_SIZE",
        description="Fallback dense vector size; auto-detected from endpoint at startup"
    )
    dense_embedding_url: str = Field(
        default="http://vllm-embedding:9091/v1", alias="DENSE_EMBEDDING_URL"
    )

    # ColBERT Embedding Configuration
    colbert_model_name: str = Field(
        default="VAGOsolutions/SauerkrautLM-Multi-ModernColBERT", alias="COLBERT_MODEL_NAME"
    )
    colbert_embedding_url: Optional[str] = Field(
        default=None,
        alias="COLBERT_EMBEDDING_URL",
        description="Optional ColBERT endpoint for late-interaction reranking",
    )

    # LiteLLM Configuration
    litellm_base_url: str = Field(
        default="http://localhost:4000/v1", alias="LITELLM_BASE_URL"
    )
    litellm_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Admin API Configuration
    qdrant_proxy_admin_key: str = Field(default="", alias="QDRANT_PROXY_ADMIN_KEY")

    # Server Configuration
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="warning", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        populate_by_name = True

    @property
    def colbert_endpoint_configured(self) -> bool:
        """Return True when a non-empty ColBERT endpoint is configured."""
        return bool((self.colbert_embedding_url or "").strip())


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Use this function for dependency injection in FastAPI routes.
    """
    return Settings()


# Convenience instance for direct imports
settings = get_settings()
