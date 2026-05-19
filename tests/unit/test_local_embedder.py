from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from nnm.config import Settings
from nnm.infra.local_embedder import LocalEmbedder


@pytest.fixture
def s(monkeypatch):
    monkeypatch.setenv("NNM_DB_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("NNM_S3_BUCKET", "b")
    from nnm.config import get_settings
    get_settings.cache_clear()
    return Settings()


@pytest.mark.asyncio
async def test_embed_returns_payload(s):
    fake_model = MagicMock()
    fake_model.encode.return_value = {
        "dense_vecs": np.zeros((2, 1024), dtype=np.float32),
        "lexical_weights": [{0: 0.1}, {1: 0.2}],
        "colbert_vecs": [
            np.zeros((5, 1024), dtype=np.float32),
            np.zeros((6, 1024), dtype=np.float32),
        ],
    }
    with patch("nnm.infra.local_embedder.BGEM3FlagModel", return_value=fake_model):
        emb = LocalEmbedder(settings=s)
        emb.load()
        out = await emb.embed(["a", "b"])

    assert out.dense.shape == (2, 1024)
    assert len(out.sparse) == 2
    assert out.colbert[0].shape == (5, 1024)


@pytest.mark.asyncio
async def test_load_idempotent(s):
    with patch("nnm.infra.local_embedder.BGEM3FlagModel") as cls:
        cls.return_value = MagicMock()
        emb = LocalEmbedder(settings=s)
        emb.load()
        emb.load()
        cls.assert_called_once()
