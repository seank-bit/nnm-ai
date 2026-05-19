from __future__ import annotations
import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PaperMeta:
    s3_key: str
    file_hash: str
    external_id: uuid.UUID | None = None
    title: str | None = None
    authors: tuple[str, ...] | None = None
    abstract: str | None = None
    venue: str | None = None
    published_year: int | None = None
    language: str | None = None
    page_count: int | None = None
    raw_json_path: str | None = None
    raw_md_path: str | None = None
