from __future__ import annotations
import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import structlog

from nnm.errors import PdfExtractionError

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class PdfExtraction:
    json_path: Path
    md_path: Path
    json_doc: dict
    markdown: str


@dataclass
class PdfExtractor:
    storage_root: Path
    threads: int = 4

    async def _run_opendataloader(
        self, pdf_path: Path, out_dir: Path, file_hash: str
    ) -> tuple[Path, Path]:
        cmd = [
            "opendataloader-pdf",
            "-o", str(out_dir),
            "-f", "json,markdown",
            "--reading-order", "xycut",
            "--use-struct-tree",
            "--threads", str(self.threads),
            "--table-method", "cluster",
            "--image-output", "external",
            "--image-dir", str(out_dir / "images"),
            str(pdf_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise PdfExtractionError(
                f"opendataloader-pdf exit {proc.returncode}: "
                f"{stderr.decode(errors='ignore')}"
            )
        json_path = out_dir / f"{file_hash}.json"
        md_path = out_dir / f"{file_hash}.md"
        if not json_path.exists() or not md_path.exists():
            produced = sorted(p.name for p in out_dir.glob(f"{file_hash}.*"))
            raise PdfExtractionError(
                f"opendataloader-pdf output missing for {file_hash}; produced={produced}"
            )
        return json_path, md_path

    async def extract(self, pdf_bytes: bytes, file_hash: str) -> PdfExtraction:
        out_dir = self.storage_root / "extracted"
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / f"{file_hash}.pdf"
            tmp_path.write_bytes(pdf_bytes)
            json_path, md_path = await self._run_opendataloader(tmp_path, out_dir, file_hash)

        raw_doc = json.loads(json_path.read_text(encoding="utf-8"))
        json_doc = _normalize_doc(raw_doc)
        markdown = md_path.read_text(encoding="utf-8")
        log.info("pdf.extracted", file_hash=file_hash,
                 json_kb=json_path.stat().st_size // 1024,
                 elements=len(json_doc["elements"]))
        return PdfExtraction(
            json_path=json_path, md_path=md_path,
            json_doc=json_doc, markdown=markdown,
        )


def _normalize_doc(raw: dict) -> dict:
    elements: list[dict] = []
    _walk_kids(raw.get("kids", []), elements)
    return {
        "metadata": {
            "title": _decode_pdf_title(raw.get("title")),
            "author": _decode_pdf_title(raw.get("author")),
            "language": None,
        },
        "elements": elements,
    }


def _decode_pdf_title(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if not (s.startswith("<") and len(s) > 2):
        return s
    body = s[1:-1] if s.endswith(">") else s[1:]
    body = body.replace(" ", "").replace("\n", "")
    if not body or not all(c in "0123456789ABCDEFabcdef" for c in body):
        return s
    if len(body) % 2:
        body = body + "0"
    try:
        b = bytes.fromhex(body)
    except ValueError:
        return s
    if b.startswith(b"\xfe\xff"):
        return b[2:].decode("utf-16-be", errors="ignore").rstrip("\x00").strip() or None
    if b.startswith(b"\xff\xfe"):
        return b[2:].decode("utf-16-le", errors="ignore").rstrip("\x00").strip() or None
    for enc in ("cp949", "euc-kr", "utf-8"):
        try:
            decoded = b.decode(enc).rstrip("\x00").strip()
        except UnicodeDecodeError:
            continue
        if decoded and all(ch.isprintable() or ch.isspace() for ch in decoded):
            return decoded
    return b.decode("latin1", errors="ignore").rstrip("\x00").strip() or None


def _walk_kids(items, out: list[dict]) -> None:
    if not isinstance(items, list):
        return
    for it in items:
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        page = it.get("page number")
        if t == "heading":
            content = it.get("content")
            if isinstance(content, str) and content.strip():
                out.append({
                    "type": "heading",
                    "text": content,
                    "level": it.get("heading level") or it.get("level"),
                    "page": page,
                })
            continue
        content = it.get("content")
        if isinstance(content, str) and content.strip():
            out.append({"type": "paragraph", "text": content, "page": page})
        for sub_key in ("kids", "list items", "rows", "cells"):
            sub = it.get(sub_key)
            if isinstance(sub, list):
                _walk_kids(sub, out)
