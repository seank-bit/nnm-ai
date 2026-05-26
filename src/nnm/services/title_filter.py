from __future__ import annotations

_GARBAGE_TITLE_PATTERNS = (
    "epapyrus pdf document",
    "untitled",
    "untitled document",
    "untitled-1",
    "adobe photoshop pdf",
    "microsoft word - ",
    "microsoft word",
    "microsoft powerpoint - ",
    "microsoft powerpoint",
    "microsoft excel - ",
    "microsoft excel",
    "microsoft visio - ",
    "microsoft visio",
    "print",
    "document1",
    "powerpoint presentation",
    "hwp document",
    "한글과컴퓨터 한글",
)


def is_garbage_title(s: str) -> bool:
    low = s.strip().lower()
    if not low:
        return True
    if len(low) < 4:
        return True
    if low.endswith((".indd", ".dvi", ".tex", ".doc", ".docx", ".hwp", ".pdf", ".pptx", ".ppt", ".xlsx", ".xls")):
        return True
    return any(low == p or low.startswith(p) for p in _GARBAGE_TITLE_PATTERNS)


def clean_title(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or is_garbage_title(s):
        return None
    return s
