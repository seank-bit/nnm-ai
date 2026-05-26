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
from nnm.infra.ops_db import ops_session
from nnm.infra.repository import SqlBackfillRepository
from nnm.infra.s3 import S3Loader
from nnm.logging import configure_logging
from nnm.services.backfill import BackfillService
from nnm.services.chunker import Chunker
from nnm.services.maintenance import reset_database
from nnm.services.pdf_extractor import PdfExtractor
from nnm.services.publication_mapper import PublicationMapper
from nnm.services.title_filter import is_garbage_title

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


@app.command(name="hnsw-disable")
def hnsw_disable() -> None:
    """대량 적재 전 HNSW 인덱스 drop."""
    asyncio.run(_run_hnsw(enable=False))


@app.command(name="hnsw-enable")
def hnsw_enable() -> None:
    """대량 적재 후 HNSW 인덱스 재생성."""
    asyncio.run(_run_hnsw(enable=True))


@app.command()
def remap(
    all_rows: bool = typer.Option(
        False, "--all",
        help="external_id 가 이미 있는 paper 도 재매핑 (기본: NULL 인 것만).",
    ),
    overwrite_title: bool = typer.Option(
        False, "--overwrite-title",
        help="ops DB 의 publications.title 로 paper.title 을 무조건 덮어쓰기.",
    ),
    limit: int | None = typer.Option(None, help="처리 개수 제한"),
) -> None:
    """기존 papers 의 external_id / title 을 운영 DB 매핑으로 채움."""
    asyncio.run(_run_remap(all_rows=all_rows, overwrite_title=overwrite_title, limit=limit))


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

    s3_bucket = settings.s3_pdf_bucket or settings.s3_bucket
    s3 = S3Loader(bucket=s3_bucket, region=settings.s3_region)
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
    async with factory() as session, ops_session(settings) as ops:
        repo = SqlBackfillRepository(db=session)
        mapper = PublicationMapper(db=ops)
        svc = BackfillService(
            s3=s3, repo=repo, mapper=mapper,
            extractor=extractor, chunker=chunker, embedder=embedder,
            storage_root=storage_root,
        )

        job_id = await _create_job(session, job_name=job_name, prefix=prefix)

        effective_prefix = prefix or settings.s3_pdf_prefix or settings.s3_prefix
        async for key in s3.list_keys(effective_prefix):
            if not key or key.endswith("/"):
                continue
            await _enqueue_item(session, job_id, key)
            result = await svc.process_one(job_id=job_id, s3_key=key)
            log.info("backfill.iter", key=key, result=result)
            processed += 1
            if limit is not None and processed >= limit:
                break

        await _finalize_job(session, job_id, processed=processed)


async def _run_hnsw(*, enable: bool) -> None:
    factory = get_factory()
    async with factory() as session:
        if enable:
            log.info("hnsw.rebuild.start")
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS chunk_emb_dense_hnsw "
                "ON chunk_embeddings USING hnsw (dense vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            ))
            await session.commit()
            log.info("hnsw.rebuild.done")
            typer.echo("HNSW 인덱스 재생성 완료")
        else:
            log.info("hnsw.drop.start")
            await session.execute(text("DROP INDEX IF EXISTS chunk_emb_dense_hnsw"))
            await session.commit()
            log.info("hnsw.drop.done")
            typer.echo("HNSW 인덱스 drop 완료 (대량 적재 후 nnm hnsw-enable 로 복구)")


async def _run_remap(*, all_rows: bool, overwrite_title: bool, limit: int | None) -> None:
    settings = get_settings()
    factory = get_factory()

    scanned = 0
    matched = 0
    updated = 0

    async with factory() as session, ops_session(settings) as ops:
        if ops is None:
            log.error("remap.no_ops_db", message="DB_HOST/DB_DATABASE/DB_USERNAME 가 .env 에 없음")
            return
        mapper = PublicationMapper(db=ops)

        where = "" if all_rows else "WHERE external_id IS NULL"
        sql = f"SELECT id, s3_key, external_id, title FROM papers {where} ORDER BY id"
        rows = (await session.execute(text(sql))).all()
        targets = rows if limit is None else rows[:limit]

        for r in targets:
            scanned += 1
            mapping = await mapper.lookup(r.s3_key)
            if mapping is None:
                continue
            matched += 1
            existing_ok = bool(r.title) and not is_garbage_title(r.title)
            if mapping.title and (overwrite_title or not existing_ok):
                new_title = mapping.title
            elif existing_ok:
                new_title = r.title
            else:
                new_title = None
            await session.execute(
                text(
                    "UPDATE papers SET external_id = :eid, title = :title "
                    "WHERE id = :id"
                ),
                {"eid": str(mapping.publication_id), "title": new_title, "id": r.id},
            )
            updated += 1
            if updated % 100 == 0:
                await session.commit()
                log.info("remap.progress", scanned=scanned, matched=matched, updated=updated)
        await session.commit()

    log.info("remap.done", scanned=scanned, matched=matched, updated=updated)
    typer.echo(f"remap 완료: scanned={scanned} matched={matched} updated={updated}")


if __name__ == "__main__":
    app()
