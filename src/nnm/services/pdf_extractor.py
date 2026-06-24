from __future__ import annotations
import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import structlog

from nnm.errors import OcrRequiredError, PdfExtractionError

log = structlog.get_logger()

# 1차 추출 후 텍스트 총량이 이 값 미만이면 image-only / 불량 PDF 로 간주.
# 페이지번호만 (예: '-768- -769- ...'), 한 줄 abstract 만, 표/숫자만 들어간
# 케이스를 거르기 위한 임계. 정상 학술 논문은 첫 페이지에서도 200 자 훨씬 넘김.
TEXT_MIN_CHARS = 200


@dataclass(frozen=True, slots=True)
class PdfExtraction:
    # 추출 결과는 메모리에만 보관. 디스크 영속은 호출자(BackfillService) 가
    # S3 업로드로 담당. tempdir 은 extract() 종료와 함께 자동 삭제됨.
    json_bytes: bytes
    md_bytes: bytes
    json_doc: dict
    markdown: str


@dataclass
class PdfExtractor:
    threads: int = 4
    # opendataloader-pdf hybrid 백엔드. None 이면 텍스트 추출만 (스캔 PDF elements=0).
    hybrid_url: str | None = None
    hybrid_mode: str = "auto"
    hybrid_timeout_ms: int = 180000
    # subprocess 전체 wall-clock 한도. 초과 시 SIGKILL → PdfExtractionError.
    # 비정상 PDF 로 인한 호스트 OOM/무한루프 방지가 목적.
    extract_timeout_s: int = 1800
    # True 이면 1차 추출 elements=0 (= OCR 필요 PDF) 를 OCR 돌리지 않고
    # OcrRequiredError 로 즉시 raise. BackfillService 가 "skipped" 처리.
    skip_ocr: bool = False

    async def _run_opendataloader(
        self, pdf_path: Path, out_dir: Path, file_hash: str,
        *, use_hybrid: bool = False,
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
        ]
        if use_hybrid and self.hybrid_url:
            # 1차 텍스트 추출 결과 0 일 때만 호출되는 OCR fallback.
            # 모든 페이지를 백엔드로 보내 OCR 수행 (full + 서버측 force_ocr).
            cmd.extend([
                "--hybrid", "docling-fast",
                "--hybrid-mode", self.hybrid_mode,
                "--hybrid-url", self.hybrid_url,
                "--hybrid-timeout", str(self.hybrid_timeout_ms),
                "--hybrid-fallback",  # 백엔드 에러 시 Java 추출로 fallback
            ])
        cmd.append(str(pdf_path))
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.extract_timeout_s,
            )
        except asyncio.TimeoutError:
            # SIGKILL → 자식 자원 회수. wait() 로 zombie 방지.
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise PdfExtractionError(
                f"opendataloader-pdf timeout after {self.extract_timeout_s}s "
                f"(use_hybrid={use_hybrid})"
            ) from None
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
        # NOTE: raw-bytes 사전 컷 (_looks_image_only) 은 실측에서 정상 학술 PDF
        # (figure 많은 논문 등) 도 image-only 로 잘못 분류해 비활성. 대신 1차
        # opendataloader 결과의 텍스트 길이로 판정 (아래 TEXT_MIN_CHARS).

        # tempdir 안에서 추출하고 bytes 만 메모리로 가져옴. with 블록 종료 시
        # 임시 PDF + 산출물 디렉토리 통째로 삭제 → 로컬 영속 없음.
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / f"{file_hash}.pdf"
            tmp_path.write_bytes(pdf_bytes)
            out_dir = Path(td) / "out"
            out_dir.mkdir()

            # 1차: 텍스트 레이어 추출 (Java only, 빠름, 페이지당 ~1-3초).
            json_path, md_path = await self._run_opendataloader(
                tmp_path, out_dir, file_hash, use_hybrid=False,
            )
            json_bytes = json_path.read_bytes()
            md_bytes = md_path.read_bytes()
            json_doc = _normalize_doc(json.loads(json_bytes))

            # 1차 결과가 부실하면 (elements 0 또는 텍스트 총량 < 임계) image-only 로 간주.
            # 200 자 미만은 페이지번호 / 매우 짧은 abstract / 표 만 들어있는 케이스.
            total_chars = sum(
                len((el.get("text") or "")) for el in json_doc["elements"]
            )
            if len(json_doc["elements"]) == 0 or total_chars < TEXT_MIN_CHARS:
                if self.skip_ocr:
                    log.info(
                        "pdf.skip_ocr",
                        file_hash=file_hash,
                        elements=len(json_doc["elements"]),
                        total_chars=total_chars,
                        reason="text_too_short",
                    )
                    raise OcrRequiredError(
                        f"text extraction too sparse (elements="
                        f"{len(json_doc['elements'])}, chars={total_chars}); "
                        f"OCR disabled (file_hash={file_hash})"
                    )
                if self.hybrid_url:
                    log.info(
                        "pdf.retry_with_ocr",
                        file_hash=file_hash,
                        elements=len(json_doc["elements"]),
                        total_chars=total_chars,
                        reason="text_too_short",
                    )
                    json_path, md_path = await self._run_opendataloader(
                        tmp_path, out_dir, file_hash, use_hybrid=True,
                    )
                    json_bytes = json_path.read_bytes()
                    md_bytes = md_path.read_bytes()
                    json_doc = _normalize_doc(json.loads(json_bytes))

        log.info(
            "pdf.extracted",
            file_hash=file_hash,
            json_kb=len(json_bytes) // 1024,
            elements=len(json_doc["elements"]),
        )
        return PdfExtraction(
            json_bytes=json_bytes, md_bytes=md_bytes,
            json_doc=json_doc, markdown=md_bytes.decode("utf-8", errors="replace"),
        )


def _normalize_doc(raw: dict) -> dict:
    elements: list[dict] = []
    _walk_kids(raw.get("kids", []), elements)
    # PostgreSQL UTF-8 컬럼은 0x00 byte 거부.
    # PDF 추출 시 인코딩 잔재로 텍스트 사이 '\x00' 가 섞이는 케이스 있음
    # (예: 'ISSN:\x00 2383-8892'). chunks / paper.title 모두 보호.
    for el in elements:
        t = el.get("text")
        if isinstance(t, str) and "\x00" in t:
            el["text"] = t.replace("\x00", "")
    title = _decode_pdf_title(raw.get("title"))
    if isinstance(title, str) and "\x00" in title:
        title = title.replace("\x00", "")
    author = _decode_pdf_title(raw.get("author"))
    if isinstance(author, str) and "\x00" in author:
        author = author.replace("\x00", "")
    return {
        "metadata": {
            "title": title,
            "author": author,
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
