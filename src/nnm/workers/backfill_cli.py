from __future__ import annotations
import asyncio
from pathlib import Path

import structlog
import typer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import AutoTokenizer

from nnm.config import get_settings
from nnm.db.session import get_factory
from nnm.infra.local_embedder import LocalEmbedder
from nnm.infra.repository import SqlBackfillRepository
from nnm.infra.s3 import S3Loader
from nnm.logging import configure_logging
from nnm.services.backfill import BackfillService
from nnm.services.chunker import Chunker
from nnm.services.maintenance import reset_database
from nnm.services.pdf_extractor import PdfExtractor
from nnm.services.publication_mapper import PublicationMapper

app = typer.Typer(help="nnm-ai 백필 CLI")
log = structlog.get_logger()


@app.callback()
def _setup() -> None:
    configure_logging()


@app.command()
def ingest(
    prefix: str = typer.Option("", help="S3 prefix"),
    limit: int | None = typer.Option(None, help="처리 개수 제한"),
    job_name: str = typer.Option("backfill", help="ingest_jobs.job_name"),
) -> None:
    """S3에서 PDF를 가져와 청킹·임베딩·적재."""
    asyncio.run(_run_ingest(prefix=prefix, limit=limit, job_name=job_name))


@app.command(name="reset-db")
def reset_db(force: bool = typer.Option(False, "--force")) -> None:
    """모든 nnm 테이블 drop 후 alembic upgrade head."""
    asyncio.run(reset_database(force=force))


async def _create_job(session: AsyncSession, *, job_name: str, prefix: str) -> int:
    result = await session.execute(
        text("INSERT INTO ingest_jobs(job_name, s3_prefix) VALUES (:n, :p) RETURNING id"),
        {"n": job_name, "p": prefix},
    )
    job_id = result.scalar_one()
    await session.commit()
    return job_id


async def _enqueue_item(session: AsyncSession, job_id: int, s3_key: str) -> None:
    await session.execute(
        text(
            "INSERT INTO ingest_job_items(job_id, s3_key) VALUES (:j, :k) "
            "ON CONFLICT (job_id, s3_key) DO NOTHING"
        ),
        {"j": job_id, "k": s3_key},
    )
    await session.commit()


async def _finalize_job(session: AsyncSession, job_id: int, *, processed: int) -> None:
    await session.execute(
        text("UPDATE ingest_jobs SET processed = :p, finished_at = now() WHERE id = :id"),
        {"p": processed, "id": job_id},
    )
    await session.commit()


async def _run_ingest(*, prefix: str, limit: int | None, job_name: str) -> None:
    settings = get_settings()
    storage_root = Path(settings.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    s3 = S3Loader(bucket=settings.s3_bucket, region=settings.s3_region)
    extractor = PdfExtractor(storage_root=storage_root, threads=settings.pdf_threads)
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    chunker = Chunker(
        tokenizer=tokenizer,
        target_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    embedder = LocalEmbedder(settings=settings)
    embedder.load()

    factory = get_factory()
    processed = 0
    async with factory() as session:
        repo = SqlBackfillRepository(db=session)
        mapper = PublicationMapper(db=session)
        svc = BackfillService(
            s3=s3, repo=repo, mapper=mapper,
            extractor=extractor, chunker=chunker, embedder=embedder,
            storage_root=storage_root,
        )

        job_id = await _create_job(session, job_name=job_name, prefix=prefix)

        async for key in s3.list_keys(prefix or settings.s3_prefix):
            if not key.lower().endswith(".pdf"):
                continue
            await _enqueue_item(session, job_id, key)
            result = await svc.process_one(job_id=job_id, s3_key=key)
            log.info("backfill.iter", key=key, result=result)
            processed += 1
            if limit is not None and processed >= limit:
                break

        await _finalize_job(session, job_id, processed=processed)


if __name__ == "__main__":
    app()
