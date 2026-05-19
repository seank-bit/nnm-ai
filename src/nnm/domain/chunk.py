from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    seq: int
    section: str | None
    section_level: int | None
    page_from: int | None
    page_to: int | None
    token_count: int
    char_count: int
    text: str
    text_for_embed: str
    language: str | None
