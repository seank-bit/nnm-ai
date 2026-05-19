from __future__ import annotations
import pytest
from sqlalchemy import text

from nnm.services.maintenance import reset_nnm_tables


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_drops_nnm_tables(session):
    await session.execute(text(
        "CREATE TABLE IF NOT EXISTS publications (id UUID PRIMARY KEY, title TEXT)"
    ))
    await session.commit()

    await reset_nnm_tables(session)

    assert (await session.execute(text(
        "SELECT to_regclass('public.papers') IS NULL"
    ))).scalar() is True
    assert (await session.execute(text(
        "SELECT to_regclass('public.publications') IS NOT NULL"
    ))).scalar() is True
