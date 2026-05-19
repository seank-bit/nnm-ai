from __future__ import annotations
import numpy as np
import pytest

from nnm.domain.chunk import ChunkDraft
from nnm.domain.paper import PaperMeta
from nnm.infra.repository import SqlBackfillRepository


@pytest.mark.integration
@pytest.mark.asyncio
async def test_insert_paper_and_find_by_hash(session):
    repo = SqlBackfillRepository(db=session)
    assert await repo.find_paper_by_hash("h" * 64) is None
    pid = await repo.insert_paper(
        PaperMeta(s3_key="papers/a.pdf", file_hash="h" * 64, title="T")
    )
    assert pid > 0
    assert await repo.find_paper_by_hash("h" * 64) == pid


@pytest.mark.integration
@pytest.mark.asyncio
async def test_insert_chunks_and_embeddings(session):
    repo = SqlBackfillRepository(db=session)
    pid = await repo.insert_paper(
        PaperMeta(s3_key="papers/b.pdf", file_hash="b" * 64)
    )
    drafts = [
        ChunkDraft(
            seq=0, section="Intro", section_level=1, page_from=1, page_to=1,
            token_count=10, char_count=30,
            text="x", text_for_embed="[T] [Intro] x", language="en",
        ),
        ChunkDraft(
            seq=1, section="Methods", section_level=1, page_from=2, page_to=2,
            token_count=20, char_count=60,
            text="y", text_for_embed="[T] [Methods] y", language="en",
        ),
    ]
    ids = await repo.insert_chunks(pid, drafts)
    assert len(ids) == 2

    await repo.insert_embeddings(
        chunk_ids=ids,
        dense=np.ones((2, 1024), dtype=np.float32),
        sparse=[{0: 0.5}, {1: 0.7}],
        colbert_paths=["colbert/1.npy", "colbert/2.npy"],
    )
