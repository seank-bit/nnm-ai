from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from nnm.config import Settings, get_settings
from nnm.db.session import get_session
from nnm.infra.local_embedder import LocalEmbedder

SettingsDep = Annotated[Settings, Depends(get_settings)]


async def _session_dep() -> AsyncIterator[AsyncSession]:
    async for s in get_session():
        yield s


DbDep = Annotated[AsyncSession, Depends(_session_dep)]


def get_embedder(request: Request) -> LocalEmbedder:
    return request.app.state.embedder  # type: ignore[no-any-return]


EmbedderDep = Annotated[LocalEmbedder, Depends(get_embedder)]
