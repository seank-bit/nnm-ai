from __future__ import annotations
import numpy as np
import pytest

from nnm.domain.chunk import ChunkDraft
from nnm.domain.embedding import EmbeddingPayload
from nnm.domain.paper import PaperMeta


def test_paper_meta_immutable():
    m = PaperMeta(s3_key="papers/a.pdf", file_hash="h" * 64)
    with pytest.raises(Exception):
        m.title = "x"  # type: ignore[misc]


def test_paper_meta_defaults():
    m = PaperMeta(s3_key="k", file_hash="h" * 64)
    assert m.title is None and m.external_id is None


def test_chunk_draft_prefix_shape():
    c = ChunkDraft(
        seq=0, section="Intro", section_level=1, page_from=1, page_to=1,
        token_count=120, char_count=600,
        text="raw body", text_for_embed="[Title] [Intro] raw body", language="ko",
    )
    assert c.token_count == 120
    assert c.text_for_embed.startswith("[Title]")


def test_embedding_payload_shapes():
    p = EmbeddingPayload(
        dense=np.zeros((1, 1024), dtype=np.float32),
        sparse=[{0: 0.1}],
        colbert=[np.zeros((10, 1024), dtype=np.float32)],
    )
    assert p.dense.shape == (1, 1024)


def test_embedding_payload_length_mismatch_raises():
    with pytest.raises(ValueError):
        EmbeddingPayload(
            dense=np.zeros((2, 1024), dtype=np.float32),
            sparse=[{0: 0.1}],
            colbert=[np.zeros((1, 1024), dtype=np.float32)],
        )
