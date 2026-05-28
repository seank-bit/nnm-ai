from __future__ import annotations
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from nnm.services.title_filter import clean_title

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class PublicationMapping:
    publication_id: uuid.UUID
    title: str | None
    legacy_aid: int | None = None


def _normalize_s3_key(s3_key: str) -> str:
    name = s3_key.rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


@dataclass
class PublicationMapper:
    csv_path: Path | None
    encoding: str = "cp949"
    _index: dict[str, PublicationMapping] = field(default_factory=dict)
    _loaded: bool = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.csv_path is None:
            log.warning("publication_mapper.csv_not_configured")
            return
        if not self.csv_path.exists():
            log.warning("publication_mapper.csv_missing", path=str(self.csv_path))
            return
        loaded = 0
        skipped = 0
        # errors="replace": CP949 export 에 깨진 바이트가 있어도 파이프라인이 멈추지 않도록.
        with self.csv_path.open(
            "r", encoding=self.encoding, errors="replace", newline="",
        ) as f:
            reader = csv.DictReader(f)
            for row in reader:
                s3_key = (row.get("s3_key") or "").strip()
                raw_id = (row.get("id") or "").strip()
                if not s3_key or not raw_id:
                    skipped += 1
                    continue
                try:
                    pid = uuid.UUID(raw_id)
                except ValueError:
                    skipped += 1
                    continue
                raw_legacy = (row.get("legacy_aid") or "").strip()
                try:
                    legacy_aid = int(raw_legacy) if raw_legacy else None
                except ValueError:
                    legacy_aid = None
                self._index[s3_key] = PublicationMapping(
                    publication_id=pid,
                    title=clean_title(row.get("title")),
                    legacy_aid=legacy_aid,
                )
                loaded += 1
        log.info(
            "publication_mapper.loaded",
            path=str(self.csv_path), entries=loaded, skipped=skipped,
        )

    def lookup_sync(self, s3_key: str) -> PublicationMapping | None:
        """동기 버전 lookup. 정렬 등 일괄처리에서 사용."""
        self._load()
        if not self._index:
            return None
        m = self._index.get(s3_key)
        if m is not None:
            return m
        stem = _normalize_s3_key(s3_key)
        return self._index.get(stem)

    async def lookup(self, s3_key: str) -> PublicationMapping | None:
        self._load()
        if not self._index:
            return None
        m = self._index.get(s3_key)
        if m is not None:
            return m
        stem = _normalize_s3_key(s3_key)
        m = self._index.get(stem)
        if m is None:
            log.warning("publication_mapper.miss", s3_key=s3_key, stem=stem)
        return m

    def legacy_aid_for(self, s3_key: str) -> int | None:
        """정렬용. CSV에 없거나 legacy_aid가 비면 None."""
        m = self.lookup_sync(s3_key)
        return m.legacy_aid if m else None
