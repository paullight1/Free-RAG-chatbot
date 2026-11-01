"""Environment configuration. API keys are required; no anonymous fallback."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tenant_api_keys_json: str = "{}"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "industry_rag_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = ""  # Empty = retrieval-only mode, no paid API necessary
    llm_api_key: str = ""
    llm_base_url: str = ""
    max_upload_bytes: int = 10_000_000
    max_chunks_per_document: int = 500

    def tenant_keys(self) -> dict[str, str]:
        try:
            parsed: Any = json.loads(self.tenant_api_keys_json)
        except json.JSONDecodeError as exc:
            raise ValueError("TENANT_API_KEYS_JSON must be valid JSON") from exc
        if not isinstance(parsed, dict) or not all(
            isinstance(tenant, str) and tenant and isinstance(key, str) and len(key) >= 12
            for tenant, key in parsed.items()
        ):
            raise ValueError("TENANT_API_KEYS_JSON must map tenant names to keys of at least 12 characters")
        if len(set(parsed.values())) != len(parsed):
            raise ValueError("Each tenant must have its own unique API key")
        return parsed


@lru_cache
def get_settings() -> Settings:
    return Settings()
