"""Tenant-authenticated FastAPI HTTP surface for the RAG lab."""
from __future__ import annotations

import hmac
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .core import RAGService
from .settings import Settings, get_settings

app = FastAPI(title="Industry RAG Engineering Lab", version="0.1.0")


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=15)


@lru_cache
def get_service() -> RAGService:
    return RAGService(get_settings())


def authenticated_tenant(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(get_settings),
) -> str:
    try:
        keys = settings.tenant_keys()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Server authentication is not configured correctly") from exc
    if not keys:
        raise HTTPException(status_code=503, detail="Server has no configured tenant credentials")
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key is required")
    for tenant, key in keys.items():
        if hmac.compare_digest(x_api_key, key):
            return tenant
    raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}  # Liveness only; does not assert DB/model readiness.


@app.post("/v1/documents")
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF, TXT or Markdown; max 10MB")],
    tenant: str = Depends(authenticated_tenant),
    service: RAGService = Depends(get_service),
    settings: Settings = Depends(get_settings),
) -> dict:
    if not file.filename:
        raise HTTPException(400, detail="A document filename is required")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, detail="Upload too large")
    try:
        return await run_in_threadpool(service.ingest, tenant, file.filename, content)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc


@app.post("/v1/search")
def search_documents(
    query: Query,
    tenant: str = Depends(authenticated_tenant),
    service: RAGService = Depends(get_service),
) -> dict:
    hits = service.search(tenant, query.question, top_k=query.top_k)
    return {"sources": [hit.citation(i) for i, hit in enumerate(hits, 1)]}


@app.post("/v1/ask")
def ask_documents(
    query: Query,
    tenant: str = Depends(authenticated_tenant),
    service: RAGService = Depends(get_service),
) -> dict:
    return service.ask(tenant, query.question, top_k=query.top_k)
