from __future__ import annotations
import os
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nnm.db.base import Base
import nnm.db.models  # noqa: F401


@pytest.fixture(scope="session")
def db_url() -> str:
    url = os.environ.get("NNM_TEST_DB_URL")
    if not url:
        pytest.skip("NNM_TEST_DB_URL not set")
    return url


@pytest_asyncio.fixture
async def engine(db_url):
    eng = create_async_engine(db_url, pool_pre_ping=True)
    async with eng.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.exec_driver_sql("DROP SCHEMA public CASCADE")
        await conn.exec_driver_sql("CREATE SCHEMA public")
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
