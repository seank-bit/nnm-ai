from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import structlog

from nnm.domain.chunk import ChunkDraft

log = structlog.get_logger()

_REFERENCE_HEADINGS = {"references", "참고문헌", "참 고 문 헌", "bibliography"}


def _split_recursive(
    text: str, target_tokens: int, overlap_tokens: int, tokenizer: Any
) -> list[str]:
    if not text.strip():
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= target_tokens:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]

    def _split_one(t: str) -> list[str]:
        for sep in separators:
            if sep in t:
                parts = [p for p in t.split(sep) if p.strip()]
                if len(parts) > 1:
                    return parts
        n = max(1, len(tokenizer.encode(t, add_special_tokens=False)) // target_tokens + 1)
        size = max(1, len(t) // n)
        return [t[i:i + size] for i in range(0, len(t), size)]

    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for part in _split_one(text):
        pt = len(tokenizer.encode(part, add_special_tokens=False))
        if pt > target_tokens:
            for sub in _split_recursive(part, target_tokens, overlap_tokens, tokenizer):
                pieces.append(sub)
            continue
        if buf_tokens + pt > target_tokens:
            pieces.append(" ".join(buf))
            overlap: list[str] = []
            ot = 0
            for prev in reversed(buf):
                pt2 = len(tokenizer.encode(prev, add_special_tokens=False))
                if ot + pt2 > overlap_tokens:
                    break
                overlap.insert(0, prev)
                ot += pt2
            buf = overlap + [part]
            buf_tokens = ot + pt
        else:
            buf.append(part)
            buf_tokens += pt
    if buf:
        pieces.append(" ".join(buf))
    return [p.strip() for p in pieces if p.strip()]


@dataclass
class Chunker:
    tokenizer: Any
    target_tokens: int = 512
    overlap_tokens: int = 64

    def chunk(
        self, doc: dict, *, paper_title: str | None, language: str | None = None
    ) -> Iterator[ChunkDraft]:
        sections = _group_sections(doc.get("elements", []))
        seq = 0
        for section in sections:
            name = (section["heading"] or "").strip()
            if name.lower() in _REFERENCE_HEADINGS:
                continue
            text = section["text"]
            if not text.strip():
                continue
            for sub in _split_recursive(
                text, self.target_tokens, self.overlap_tokens, self.tokenizer
            ):
                tokens = self.tokenizer.encode(sub, add_special_tokens=False)
                if paper_title and name:
                    prefix = f"[{paper_title}] [{name}]"
                elif name:
                    prefix = f"[{name}]"
                else:
                    prefix = ""
                text_for_embed = f"{prefix} {sub}".strip()
                yield ChunkDraft(
                    seq=seq,
                    section=name or None,
                    section_level=section.get("level"),
                    page_from=section.get("page_from"),
                    page_to=section.get("page_to"),
                    token_count=len(tokens),
                    char_count=len(sub),
                    text=sub,
                    text_for_embed=text_for_embed,
                    language=language,
                )
                seq += 1


def _group_sections(elements: list[dict]) -> list[dict]:
    sections: list[dict] = []
    current: dict | None = None

    def _new(heading: str | None, level: int | None, page: int | None) -> dict:
        return {"heading": heading, "level": level, "text": "",
                "page_from": page, "page_to": page}

    for el in elements:
        kind = el.get("type")
        page = el.get("page")
        if kind == "heading":
            if current:
                sections.append(current)
            current = _new(el.get("text", ""), el.get("level"), page)
        else:
            if current is None:
                current = _new(None, None, page)
            text = el.get("text") or ""
            if text:
                current["text"] += text + "\n"
                if page is not None:
                    current["page_to"] = page
                    if current["page_from"] is None:
                        current["page_from"] = page
    if current:
        sections.append(current)
    return sections
