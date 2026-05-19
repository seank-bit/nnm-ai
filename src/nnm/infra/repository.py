from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nnm.db.models import Chunk, ChunkEmbedding, IngestJobItem, Paper
from nnm.domain.chunk import ChunkDraft
from nnm.domain.paper import PaperMeta

log = structlog.get_logger()


@dataclass
class SqlBackfillRepository:
    db: AsyncSession
    model_name: str = "bge-m3"
    model_version: str = "v1"

    async def find_paper_by_hash(self, file_hash: str) -> int | None:
        result = await self.db.execute(
            select(Paper.id).where(Paper.file_hash == file_hash)
        )
        row = result.first()
        return row[0] if row else None

    async def insert_paper(self, meta: PaperMeta) -> int:
        paper = Paper(
            s3_key=meta.s3_key, file_hash=meta.file_hash,
            external_id=meta.external_id, title=meta.title,
            authors=list(meta.authors) if meta.authors else None,
            abstract=meta.abstract, venue=meta.venue,
            published_year=meta.published_year, language=meta.language,
            page_count=meta.page_count,
            raw_json_path=meta.raw_json_path, raw_md_path=meta.raw_md_path,
            status="extracted",
        )
        self.db.add(paper)
        await self.db.flush()
        await self.db.commit()
        return paper.id

    async def insert_chunks(self, paper_id: int, drafts: list[ChunkDraft]) -> list[int]:
        rows = [
            Chunk(
                paper_id=paper_id, seq=d.seq, section=d.section,
                section_level=d.section_level, page_from=d.page_from, page_to=d.page_to,
                token_count=d.token_count, char_count=d.char_count,
                text=d.text, text_for_embed=d.text_for_embed, language=d.language,
            )
            for d in drafts
        ]
        self.db.add_all(rows)
        await self.db.flush()
        await self.db.commit()
        return [r.id for r in rows]

    async def insert_embeddings(
        self, *, chunk_ids: list[int],
        dense: np.ndarray, sparse: list[dict[int, float]],
        colbert_paths: list[str | None],
    ) -> None:
        rows = [
            ChunkEmbedding(
                chunk_id=cid, model_name=self.model_name, model_version=self.model_version,
                dense=dense[i].tolist(),
                sparse={str(k): v for k, v in sparse[i].items()} if sparse[i] else None,
                colbert_path=colbert_paths[i],
            )
            for i, cid in enumerate(chunk_ids)
        ]
        self.db.add_all(rows)
        await self.db.commit()

    async def mark_item_done(self, job_id: int, s3_key: str, *, paper_id: int) -> None:
        await self.db.execute(
            update(IngestJobItem)
            .where(IngestJobItem.job_id == job_id, IngestJobItem.s3_key == s3_key)
            .values(status="embedded", paper_id=paper_id, error=None)
        )
        await self.db.commit()

    async def mark_item_failed(self, job_id: int, s3_key: str, reason: str) -> None:
        await self.db.execute(
            update(IngestJobItem)
            .where(IngestJobItem.job_id == job_id, IngestJobItem.s3_key == s3_key)
            .values(status="failed", error=reason[:1000])
        )
        await self.db.commit()
