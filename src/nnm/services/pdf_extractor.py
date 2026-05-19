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
        json_path = out_dir / f"{file_hash}.json"
        md_path = out_dir / f"{file_hash}.md"
        cmd = [
            "opendataloader-pdf",
            "--format", "json,markdown",
            "--reading-order", "xycut",
            "--use-struct-tree",
            "--threads", str(self.threads),
            "--table-method", "cluster",
            "--image-output", "external",
            "--image-dir", str(out_dir / "images"),
            "--output-json", str(json_path),
            "--output-markdown", str(md_path),
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
        return json_path, md_path

    async def extract(self, pdf_bytes: bytes, file_hash: str) -> PdfExtraction:
        out_dir = self.storage_root / "extracted"
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        try:
            json_path, md_path = await self._run_opendataloader(tmp_path, out_dir, file_hash)
        finally:
            tmp_path.unlink(missing_ok=True)

        json_doc = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = md_path.read_text(encoding="utf-8")
        log.info("pdf.extracted", file_hash=file_hash,
                 json_kb=json_path.stat().st_size // 1024)
        return PdfExtraction(
            json_path=json_path, md_path=md_path,
            json_doc=json_doc, markdown=markdown,
        )
