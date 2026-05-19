from __future__ import annotations
import pytest
from nnm.config import Settings, get_settings


def test_settings_required_db_url(monkeypatch):
    monkeypatch.delenv("NNM_DB_URL", raising=False)
    with pytest.raises(ValueError):
        Settings(s3_bucket="bucket")  # type: ignore[call-arg]


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("NNM_DB_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("NNM_S3_BUCKET", "test-bucket")
    s = Settings()
    assert s.env == "local"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_device == "cpu"
    assert s.embedding_fp16 is False
    assert s.embedding_batch_size == 4
    assert s.embedding_max_tokens == 8192
    assert s.chunk_size_tokens == 512
    assert s.chunk_overlap_tokens == 64
    assert s.storage_root == "var"


def test_get_settings_cached(monkeypatch):
    monkeypatch.setenv("NNM_DB_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("NNM_S3_BUCKET", "test-bucket")
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
