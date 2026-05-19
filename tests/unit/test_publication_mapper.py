from __future__ import annotations
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from nnm.services.publication_mapper import PublicationMapping, PublicationMapper


@pytest.mark.asyncio
async def test_mapper_returns_mapping():
    pub_id = uuid.uuid4()
    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(first=lambda: (pub_id, "딥러닝 OO")))
    m = PublicationMapper(db=db)
    result = await m.lookup("papers/a.pdf")
    assert result == PublicationMapping(publication_id=pub_id, title="딥러닝 OO")


@pytest.mark.asyncio
async def test_mapper_returns_none_when_absent():
    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(first=lambda: None))
    m = PublicationMapper(db=db)
    assert await m.lookup("papers/missing.pdf") is None
