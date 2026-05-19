from __future__ import annotations
import asyncio

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nnm.errors import NnmError

log = structlog.get_logger()

NNM_TABLES = (
    "chunk_embeddings", "chunks", "paper_references",
    "ingest_job_items", "ingest_jobs", "papers", "alembic_version",
)


async def reset_nnm_tables(db: AsyncSession) -> None:
    """운영 테이블은 건드리지 않고 nnm 관리 테이블만 drop."""
    for t in NNM_TABLES:
        await db.execute(text(f"DROP TABLE IF EXISTS {t} CASCADE"))
    await db.commit()
    log.info("maintenance.tables_dropped", tables=NNM_TABLES)


async def reset_database(force: bool = False) -> None:
    if not force:
        raise NnmError("Refusing reset_db without --force")

    from nnm.db.session import get_factory
    factory = get_factory()
    async with factory() as session:
        await reset_nnm_tables(session)

    proc = await asyncio.create_subprocess_exec(
        "alembic", "upgrade", "head",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise NnmError(f"alembic upgrade failed: {stderr.decode(errors='ignore')}")
    log.info("maintenance.reset_done")
