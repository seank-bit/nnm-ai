from __future__ import annotations
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nnm.config import Settings


def build_ops_db_url(s: Settings) -> str | None:
    if not (s.ops_db_host and s.ops_db_database and s.ops_db_username):
        return None
    pw = s.ops_db_password or ""
    return (
        f"mysql+asyncmy://{s.ops_db_username}:{pw}"
        f"@{s.ops_db_host}:{s.ops_db_port}/{s.ops_db_database}"
        "?charset=utf8mb4"
    )


def build_ops_engine(s: Settings) -> AsyncEngine | None:
    url = build_ops_db_url(s)
    if not url:
        return None
    return create_async_engine(url, pool_pre_ping=True, pool_recycle=1800)


@asynccontextmanager
async def ops_session(s: Settings) -> AsyncIterator[AsyncSession | None]:
    engine = build_ops_engine(s)
    if engine is None:
        yield None
        return
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with factory() as session:
            yield session
    finally:
        await engine.dispose()
