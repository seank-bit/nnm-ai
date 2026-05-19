from __future__ import annotations
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class PublicationMapping:
    publication_id: uuid.UUID
    title: str | None


@dataclass
class PublicationMapper:
    db: AsyncSession

    async def lookup(self, s3_key: str) -> PublicationMapping | None:
        result = await self.db.execute(
            text(
                "SELECT pf.publication_id, p.title "
                "FROM publication_files pf "
                "JOIN publications p ON p.id = pf.publication_id "
                "WHERE pf.s3_key = :s3_key"
            ),
            {"s3_key": s3_key},
        )
        row = result.first()
        if row is None:
            log.warning("publication_mapper.miss", s3_key=s3_key)
            return None
        return PublicationMapping(publication_id=row[0], title=row[1])
