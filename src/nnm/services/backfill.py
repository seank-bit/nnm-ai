from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import structlog

from nnm.domain.paper import PaperMeta
from nnm.errors import NnmError, OcrRequiredError
from nnm.infra.local_embedder import LocalEmbedder
from nnm.infra.s3 import S3Loader
from nnm.services.chunker import Chunker
from nnm.services.pdf_extractor import PdfExtraction, PdfExtractor
from nnm.services.publication_mapper import PublicationMapper
from nnm.services.repository import BackfillRepository
from nnm.services.title_filter import is_garbage_title

log = structlog.get_logger()
_failed_record_lock = asyncio.Lock()


@dataclass
class BackfillService:
    s3: S3Loader
    repo: BackfillRepository
    mapper: PublicationMapper
    extractor: PdfExtractor
    chunker: Chunker
    embedder: LocalEmbedder
    storage_root: Path
    # extracted 결과 (.json, .md) 를 S3 에 보관하기 위한 별도 클라이언트.
    # None 이면 S3 업로드 안 함 (로컬 var/extracted 만 유지).
    extracted_uploader: S3Loader | None = None
    extracted_prefix: str = ""
    # 추출 단계 실패 (timeout / subprocess 비정상 종료 등) PDF 를 한 줄씩 append.
    # 사이즈 제한은 두지 않고, 일단 skip 한 뒤 나중에 일괄 재처리할 수 있도록 기록만 남김.
    failed_record_path: Path | None = None

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
        except OcrRequiredError as e:
            # OCR 필요한 PDF: paper row 생성 없이 item만 skipped로 마킹.
            await self.repo.mark_item_skipped(job_id, s3_key, f"ocr_required: {e}")
            log.info("backfill.skipped_ocr_required", s3_key=s3_key, file_hash=digest)
            return "skipped"
        except NnmError as e:
            await self.repo.mark_item_failed(job_id, s3_key, str(e))
            await self._record_failed_pdf(
                s3_key=s3_key, file_hash=digest, file_size=len(data), error=e,
            )
            return "failed"

        title = title or _fallback_title(extraction.json_doc)
        language = _detect_language(extraction.json_doc)

        # 추출 결과 (.json, .md) 를 S3 에 보관 (유일한 영속 위치).
        # 파일명 = 원본 PDF s3_key 의 basename. 실패 시 path=None.
        json_s3_key, md_s3_key = await self._upload_extracted(s3_key, extraction)

        meta = PaperMeta(
            s3_key=s3_key, file_hash=digest,
            external_id=external_id, title=title, language=language,
            raw_json_path=json_s3_key,
            raw_md_path=md_s3_key,
            page_count=_count_pages(extraction.json_doc),
        )
        paper_id = await self.repo.insert_paper(meta)

        drafts = list(self.chunker.chunk(
            extraction.json_doc, paper_title=title, language=language,
        ))
        if not drafts:
            await self.repo.mark_item_done(job_id, s3_key, paper_id=paper_id)
            return "ok"

        try:
            chunk_ids = await self.repo.insert_chunks(paper_id, drafts)
            settings = self.embedder.settings
            payload = await self.embedder.embed(
                [d.text_for_embed for d in drafts],
                return_dense=True,
                return_sparse=settings.embedding_sparse,
                return_colbert=settings.embedding_colbert,
            )

            colbert_paths: list[str | None] = [None] * len(chunk_ids)
            if settings.embedding_colbert:
                colbert_dir = self.storage_root / "colbert"
                colbert_dir.mkdir(parents=True, exist_ok=True)
                for i, (cid, vec) in enumerate(zip(chunk_ids, payload.colbert)):
                    if vec is None:
                        continue
                    p = colbert_dir / f"{cid}.npy"
                    np.save(p, vec)
                    colbert_paths[i] = str(p.relative_to(self.storage_root))

            await self.repo.insert_embeddings(
                chunk_ids=chunk_ids, dense=payload.dense,
                sparse=payload.sparse, colbert_paths=colbert_paths,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("backfill.partial_failure", s3_key=s3_key, paper_id=paper_id, error=str(e))
            await self.repo.delete_paper(paper_id)
            await self.repo.mark_item_failed(job_id, s3_key, f"post-insert failure: {e}")
            return "failed"
        await self.repo.mark_item_done(job_id, s3_key, paper_id=paper_id)
        log.info("backfill.ok", s3_key=s3_key, paper_id=paper_id, chunks=len(drafts))
        return "ok"


    async def _record_failed_pdf(
        self, *, s3_key: str, file_hash: str, file_size: int, error: Exception,
    ) -> None:
        """추출 실패 PDF 를 JSONL 한 줄로 기록. 경로 미설정/IO 실패 시 silent.

        JSON 기록이 ingest 루프를 막지 않도록 모든 예외를 흡수한다.
        OOM/timeout/대용량 PDF 등 사이즈로 거를 수 없는 케이스를 누적해두고
        나중에 일괄 재처리하는 용도.
        """
        if self.failed_record_path is None:
            return
        entry = {
            "s3_key": s3_key,
            "file_hash": file_hash,
            "file_size_bytes": file_size,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            path = self.failed_record_path
            path.parent.mkdir(parents=True, exist_ok=True)
            # 동일 프로세스 내 동시성에 대비. 다중 프로세스라면 append (O_APPEND) 가
            # POSIX 상 line-atomic 이므로 이 정도면 충분.
            async with _failed_record_lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.info(
                "backfill.failed_pdf_recorded",
                s3_key=s3_key, file_size=file_size, path=str(path),
            )
        except Exception as rec_err:  # noqa: BLE001
            log.warning(
                "backfill.failed_pdf_record_error",
                s3_key=s3_key, error=str(rec_err),
            )

    async def _upload_extracted(
        self, s3_key: str, extraction: PdfExtraction,
    ) -> tuple[str | None, str | None]:
        """추출 결과를 S3 에 업로드하고 (json_key, md_key) 반환.

        uploader 미설정/업로드 실패 시 (None, None) — paper 는 path 컬럼 NULL 로 들어감.
        """
        if self.extracted_uploader is None:
            return None, None
        # PDF s3_key: 'newnonmuncom-pdf/0002f0...' → basename '0002f0...'
        basename = s3_key.rsplit("/", 1)[-1]
        if "." in basename:
            basename = basename.rsplit(".", 1)[0]

        json_key = f"{self.extracted_prefix}{basename}.json"
        md_key = f"{self.extracted_prefix}{basename}.md"

        try:
            await self.extracted_uploader.upload_bytes(
                json_key, extraction.json_bytes, content_type="application/json",
            )
            await self.extracted_uploader.upload_bytes(
                md_key, extraction.md_bytes, content_type="text/markdown; charset=utf-8",
            )
            log.info(
                "extracted.uploaded",
                s3_key=s3_key, json_key=json_key, md_key=md_key,
                json_bytes=len(extraction.json_bytes), md_bytes=len(extraction.md_bytes),
            )
            return json_key, md_key
        except Exception as e:  # noqa: BLE001 — 업로드 실패는 ingest 차단 아님
            log.warning(
                "extracted.upload_failed",
                s3_key=s3_key, json_key=json_key, md_key=md_key, error=str(e),
            )
            return None, None


def _fallback_title(doc: dict) -> str | None:
    meta = doc.get("metadata") or {}
    raw = meta.get("title")
    if raw:
        cleaned = str(raw).strip()
        if cleaned and not is_garbage_title(cleaned):
            return cleaned
    elements = doc.get("elements") or []
    for el in elements:
        if el.get("type") == "heading" and el.get("level") == 1:
            text = str(el.get("text", "")).strip()
            if text and len(text) >= 4:
                return text
    for el in elements:
        if el.get("type") == "heading":
            text = str(el.get("text", "")).strip()
            if text and len(text) >= 4:
                return text
    return None


def _detect_language(doc: dict) -> str | None:
    meta = doc.get("metadata") or {}
    lang = meta.get("language")
    return str(lang) if lang else None


def _count_pages(doc: dict) -> int | None:
    pages = {el.get("page") for el in doc.get("elements", []) if el.get("page")}
    return max(pages) if pages else None
