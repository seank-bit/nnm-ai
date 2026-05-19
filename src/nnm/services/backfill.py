from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import structlog

from nnm.domain.paper import PaperMeta
from nnm.errors import NnmError
from nnm.infra.local_embedder import LocalEmbedder
from nnm.infra.s3 import S3Loader
from nnm.services.chunker import Chunker
from nnm.services.pdf_extractor import PdfExtractor
from nnm.services.publication_mapper import PublicationMapper
from nnm.services.repository import BackfillRepository

log = structlog.get_logger()


@dataclass
class BackfillService:
    s3: S3Loader
    repo: BackfillRepository
    mapper: PublicationMapper
    extractor: PdfExtractor
    chunker: Chunker
    embedder: LocalEmbedder
    storage_root: Path

    async def process_one(
        self, *, job_id: int, s3_key: str,
    ) -> Literal["ok", "skipped", "failed"]:
        try:
            data, digest = await self.s3.download(s3_key)
        except NnmError as e:
            await self.repo.mark_item_failed(job_id, s3_key, str(e))
            return "failed"

        existing = await self.repo.find_paper_by_hash(digest)
        if existing is not None:
            log.info("backfill.skipped", s3_key=s3_key, paper_id=existing)
            await self.repo.mark_item_done(job_id, s3_key, paper_id=existing)
            return "skipped"

        mapping = await self.mapper.lookup(s3_key)
        title = mapping.title if mapping else None
        external_id = mapping.publication_id if mapping else None

        try:
            extraction = await self.extractor.extract(data, file_hash=digest)
        except NnmError as e:
            await self.repo.mark_item_failed(job_id, s3_key, str(e))
            return "failed"

        title = title or _fallback_title(extraction.json_doc)
        language = _detect_language(extraction.json_doc)

        meta = PaperMeta(
            s3_key=s3_key, file_hash=digest,
            external_id=external_id, title=title, language=language,
            raw_json_path=str(extraction.json_path.relative_to(self.storage_root)),
            raw_md_path=str(extraction.md_path.relative_to(self.storage_root)),
            page_count=_count_pages(extraction.json_doc),
        )
        paper_id = await self.repo.insert_paper(meta)

        drafts = list(self.chunker.chunk(
            extraction.json_doc, paper_title=title, language=language,
        ))
        if not drafts:
            await self.repo.mark_item_done(job_id, s3_key, paper_id=paper_id)
            return "ok"

        chunk_ids = await self.repo.insert_chunks(paper_id, drafts)
        payload = await self.embedder.embed([d.text_for_embed for d in drafts])

        colbert_dir = self.storage_root / "colbert"
        colbert_dir.mkdir(parents=True, exist_ok=True)
        colbert_paths: list[str | None] = []
        for cid, vec in zip(chunk_ids, payload.colbert):
            if vec is None:
                colbert_paths.append(None)
                continue
            p = colbert_dir / f"{cid}.npy"
            np.save(p, vec)
            colbert_paths.append(str(p.relative_to(self.storage_root)))

        await self.repo.insert_embeddings(
            chunk_ids=chunk_ids, dense=payload.dense,
            sparse=payload.sparse, colbert_paths=colbert_paths,
        )
        await self.repo.mark_item_done(job_id, s3_key, paper_id=paper_id)
        log.info("backfill.ok", s3_key=s3_key, paper_id=paper_id, chunks=len(drafts))
        return "ok"


def _fallback_title(doc: dict) -> str | None:
    meta = doc.get("metadata") or {}
    if meta.get("title"):
        return str(meta["title"]).strip() or None
    for el in doc.get("elements", []):
        if el.get("type") == "heading" and el.get("level") == 1:
            return str(el.get("text", "")).strip() or None
    return None


def _detect_language(doc: dict) -> str | None:
    meta = doc.get("metadata") or {}
    lang = meta.get("language")
    return str(lang) if lang else None


def _count_pages(doc: dict) -> int | None:
    pages = {el.get("page") for el in doc.get("elements", []) if el.get("page")}
    return max(pages) if pages else None
