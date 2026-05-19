from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("NNM_DB_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("NNM_S3_BUCKET", "b")
    from nnm.config import get_settings
    get_settings.cache_clear()


def test_healthz_ok():
    with patch("nnm.infra.local_embedder.LocalEmbedder.load"):
        from nnm.main import create_app
        client = TestClient(create_app(load_model=False))
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_correlation_id_echoed():
    with patch("nnm.infra.local_embedder.LocalEmbedder.load"):
        from nnm.main import create_app
        client = TestClient(create_app(load_model=False))
        r = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
        assert r.headers.get("X-Request-ID") == "abc-123"


def test_correlation_id_generated_when_absent():
    with patch("nnm.infra.local_embedder.LocalEmbedder.load"):
        from nnm.main import create_app
        client = TestClient(create_app(load_model=False))
        r = client.get("/healthz")
        assert "X-Request-ID" in r.headers
        assert len(r.headers["X-Request-ID"]) > 8
