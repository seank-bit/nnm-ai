from __future__ import annotations
import uuid
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nnm.domain.chunk import ChunkDraft
from nnm.domain.embedding import EmbeddingPayload
from nnm.services.backfill import BackfillService
from nnm.services.publication_mapper import PublicationMapping


def _make_chunks(n: int) -> list[ChunkDraft]:
    return [
        ChunkDraft(
            seq=i, section="Intro", section_level=1, page_from=1, page_to=1,
            token_count=100, char_count=300,
            text=f"body {i}", text_for_embed=f"[T] [Intro] body {i}", language="en",
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_skips_when_hash_known(tmp_path: Path):
    s3 = MagicMock(); s3.download = AsyncMock(return_value=(b"x", "h" * 64))
    repo = MagicMock(); repo.find_paper_by_hash = AsyncMock(return_value=42)
    repo.mark_item_done = AsyncMock()
    svc = BackfillService(
        s3=s3, repo=repo, mapper=MagicMock(), extractor=MagicMock(),
        chunker=MagicMock(), embedder=MagicMock(), storage_root=tmp_path,
    )
    assert await svc.process_one(job_id=1, s3_key="papers/x.pdf") == "skipped"


@pytest.mark.asyncio
async def test_full_flow(tmp_path: Path):
    s3 = MagicMock(); s3.download = AsyncMock(return_value=(b"x", "a" * 64))
    repo = MagicMock()
    repo.find_paper_by_hash = AsyncMock(return_value=None)
    repo.insert_paper = AsyncMock(return_value=7)
    repo.insert_chunks = AsyncMock(return_value=[101, 102])
    repo.insert_embeddings = AsyncMock()
    repo.mark_item_done = AsyncMock()

    mapper = MagicMock()
    mapper.lookup = AsyncMock(return_value=PublicationMapping(
        publication_id=uuid.uuid4(), title="딥러닝 OO",
    ))

    extractor = MagicMock()
    extraction = MagicMock()
    extraction.json_doc = {"elements": []}
    extraction.json_path = tmp_path / "x.json"
    extraction.md_path = tmp_path / "x.md"
    extraction.markdown = "# x"
    extractor.extract = AsyncMock(return_value=extraction)

    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=iter(_make_chunks(2)))

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=EmbeddingPayload(
        dense=np.zeros((2, 1024), dtype=np.float32),
        sparse=[{0: 0.1}, {0: 0.2}],
        colbert=[np.zeros((5, 1024), dtype=np.float32),
                 np.zeros((5, 1024), dtype=np.float32)],
    ))

    svc = BackfillService(
        s3=s3, repo=repo, mapper=mapper, extractor=extractor,
        chunker=chunker, embedder=embedder, storage_root=tmp_path,
    )
    assert await svc.process_one(job_id=1, s3_key="papers/x.pdf") == "ok"
    repo.insert_paper.assert_awaited_once()
    repo.insert_chunks.assert_awaited_once()
    repo.insert_embeddings.assert_awaited_once()
    repo.mark_item_done.assert_awaited_once()
