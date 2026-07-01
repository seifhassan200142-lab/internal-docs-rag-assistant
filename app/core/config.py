from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "internal-docs-rag-assistant"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    upload_dir: Path = Path("data/uploads")
    index_dir: Path = Path("data/index")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "internal_docs_chunks"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=900, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)

    retrieval_top_k: int = Field(default=6, ge=1, le=30)
    rrf_k: int = Field(default=60, ge=1)

    enable_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_provider: Literal["openai", "groq", "ollama", "openai_compatible"] = "openai"
    request_timeout_seconds: int = Field(default=60, ge=5)

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"

    openai_compatible_api_key: str | None = None
    openai_compatible_base_url: str | None = None
    openai_compatible_model: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    min_context_chunks: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
