from __future__ import annotations
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nnm.services.title_filter import clean_title

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class PublicationMapping:
    publication_id: uuid.UUID
    title: str | None


def _normalize_s3_key(s3_key: str) -> str:
    name = s3_key.rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


def _coerce_uuid(value: object) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, (bytes, bytearray)):
        b = bytes(value)
        if len(b) == 16:
            return uuid.UUID(bytes=b)
        return uuid.UUID(b.decode("ascii", errors="ignore"))
    return uuid.UUID(str(value))


@dataclass
class PublicationMapper:
    db: AsyncSession | None
    _tables_checked: bool = False
    _tables_available: bool = False

    async def _ensure_tables(self) -> bool:
        if self.db is None:
            return False
        if self._tables_checked:
            return self._tables_available
        exists = (await self.db.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name IN ('publications', 'publication_files')"
            )
        )).scalar()
        self._tables_available = int(exists or 0) >= 2
        self._tables_checked = True
        if not self._tables_available:
            log.warning("publication_mapper.tables_missing")
        return self._tables_available

    async def lookup(self, s3_key: str) -> PublicationMapping | None:
        if self.db is None or not await self._ensure_tables():
            return None
        stem = _normalize_s3_key(s3_key)
        result = await self.db.execute(
            text(
                "SELECT pf.publication_id, p.title "
                "FROM publication_files pf "
                "JOIN publications p ON p.id = pf.publication_id "
                "WHERE pf.s3_key IN (:full, :stem) "
                "LIMIT 1"
            ),
            {"full": s3_key, "stem": stem},
        )
        row = result.first()
        if row is None:
            log.warning("publication_mapper.miss", s3_key=s3_key, stem=stem)
            return None
        try:
            pid = _coerce_uuid(row[0])
        except (ValueError, TypeError) as exc:
            log.warning("publication_mapper.invalid_uuid", s3_key=s3_key, error=str(exc))
            return None
        return PublicationMapping(publication_id=pid, title=clean_title(row[1]))
