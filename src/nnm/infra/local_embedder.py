from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from nnm.config import Settings
from nnm.domain.embedding import EmbeddingPayload
from nnm.errors import EmbeddingFailure

log = structlog.get_logger()

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None  # type: ignore[assignment]


@dataclass
class LocalEmbedder:
    settings: Settings
    _model: Any | None = field(default=None, init=False)

    def load(self) -> None:
        if self._model is not None:
            return
        if BGEM3FlagModel is None:
            raise EmbeddingFailure("FlagEmbedding not installed")
        log.info(
            "embedder.loading", model=self.settings.embedding_model,
            device=self.settings.embedding_device, fp16=self.settings.embedding_fp16,
        )
        self._model = BGEM3FlagModel(
            self.settings.embedding_model,
            use_fp16=self.settings.embedding_fp16,
            devices=[self.settings.embedding_device],
        )
        log.info("embedder.loaded")

    async def embed(
        self, texts: list[str], *,
        return_dense: bool = True, return_sparse: bool = True, return_colbert: bool = True,
    ) -> EmbeddingPayload:
        if self._model is None:
            self.load()
        assert self._model is not None
        out = await asyncio.to_thread(
            self._model.encode, texts,
            batch_size=self.settings.embedding_batch_size,
            max_length=self.settings.embedding_max_tokens,
            return_dense=return_dense, return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )
        n = len(texts)
        return EmbeddingPayload(
            dense=out["dense_vecs"],
            sparse=out.get("lexical_weights") or [{}] * n,
            colbert=out.get("colbert_vecs") or [None] * n,
        )
