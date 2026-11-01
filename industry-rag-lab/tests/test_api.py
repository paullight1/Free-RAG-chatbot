from __future__ import annotations

from fastapi.testclient import TestClient

from rag_lab.api import app, get_service
from rag_lab.core import RAGService
from rag_lab.settings import Settings, get_settings
from tests.test_core import DummyEmbedder, TenantStore


def test_api_requires_authentication_and_scopes_tenants():
    settings = Settings(tenant_api_keys_json='{"alpha":"super-secret-alpha-123","beta":"super-secret-beta-456"}')
    service = RAGService(settings, embedder=DummyEmbedder(), store=TenantStore())
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_service] = lambda: service
    try:
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}
            assert client.post("/v1/search", json={"question": "priority incidents"}).status_code == 401
            assert client.post("/v1/search", headers={"X-API-Key": "invalid"},
                               json={"question": "priority incidents"}).status_code == 401
            alpha = {"X-API-Key": "super-secret-alpha-123"}
            beta = {"X-API-Key": "super-secret-beta-456"}
            upload = client.post("/v1/documents", headers=alpha, files={"file": ("sla.md", b"Priority one is 15 minutes.", "text/markdown")})
            assert upload.status_code == 200
            assert upload.json()["chunks"] == 1
            assert len(client.post("/v1/search", headers=alpha,
                                   json={"question": "priority incidents"}).json()["sources"]) == 1
            assert client.post("/v1/search", headers=beta,
                               json={"question": "priority incidents"}).json()["sources"] == []
            assert client.post("/v1/ask", headers=alpha, json={"question": "a"}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_missing_tenant_keys_fail_closed():
    app.dependency_overrides[get_settings] = lambda: Settings(tenant_api_keys_json="{}")
    try:
        with TestClient(app) as client:
            assert client.post("/v1/search", headers={"X-API-Key": "any"},
                               json={"question": "some question"}).status_code == 503
    finally:
        app.dependency_overrides.clear()
