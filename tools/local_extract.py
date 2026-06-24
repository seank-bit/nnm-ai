#!/usr/bin/env python3
"""Local PDF → .json/.md generator for nnm-ai (runs on dev/laptop, uploads to S3).

Pipeline per PDF:
  1. download from S3 PDF bucket/prefix
  2. compute SHA256 of PDF bytes
  3. opendataloader-pdf subprocess → {hash}.json / {hash}.md in a tempdir
  4. parse json, sum text chars; if < TEXT_MIN_CHARS → record as OCR-required, skip
  5. inject `"file_hash"` at JSON top level + S3 object metadata header
  6. upload .json / .md to S3 extracted bucket/prefix

Skips PDFs whose stem already has BOTH .json+.md in the extracted prefix.
Sort order: legacy_aid DESC from publications CSV (CSV-missing entries trail).
S3 original PDFs are NEVER modified or deleted — read-only input.

USAGE:
    python tools/local_extract.py --limit 5            # smoke test
    python tools/local_extract.py                       # full run
    python tools/local_extract.py --help                # show all options

REQUIREMENTS:
    Python 3.10+, `pip install boto3 python-dotenv`
    `opendataloader-pdf` binary in PATH (or pass --opendataloader-bin)
    AWS credentials via env (AWS_ACCESS_KEY_ID / SECRET) or ~/.aws/credentials

CONFIG (env vars or CLI flags):
    NNM_S3_PDF_BUCKET / NNM_S3_BUCKET     PDF source bucket
    NNM_S3_PDF_PREFIX / NNM_S3_PREFIX     PDF source prefix (e.g. newnonmuncom-pdf/)
    NNM_S3_EXTRACTED_BUCKET               output bucket (defaults to PDF bucket)
    NNM_S3_EXTRACTED_PREFIX               output prefix (default newnonmuncom-extracted/)
    NNM_S3_REGION                         AWS region (default ap-northeast-2)
    NNM_PUBLICATION_CSV                   publications_*.csv path (for legacy_aid sort)
    NNM_PUBLICATION_CSV_ENCODING          default cp949
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    sys.exit("ERROR: boto3 not installed. Run: pip install boto3")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading optional


# 1차 텍스트 추출 총량이 이 값 미만이면 image-only / OCR 필요 PDF 로 간주.
# 페이지 번호만 / 매우 짧은 abstract / 표 만 들어있는 케이스 컷.
# 서버 측 PdfExtractor 의 TEXT_MIN_CHARS 와 일치해야 동일 판정.
TEXT_MIN_CHARS = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--s3-pdf-bucket", default=None)
    p.add_argument("--s3-pdf-prefix", default=None)
    p.add_argument("--s3-extracted-bucket", default=None)
    p.add_argument("--s3-extracted-prefix", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--publication-csv", default=None)
    p.add_argument("--publication-csv-encoding", default=None)
    p.add_argument(
        "--opendataloader-bin", default="opendataloader-pdf",
        help="path to opendataloader-pdf executable",
    )
    p.add_argument("--threads", type=int, default=4, help="opendataloader threads")
    p.add_argument(
        "--extract-timeout-s", type=int, default=1800,
        help="per-PDF wall-clock limit (kill subprocess on timeout)",
    )
    p.add_argument("--limit", type=int, default=None, help="process at most N PDFs (existing-skipped not counted)")
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument(
        "--failed-log", default="failed_pdfs.jsonl",
        help="append extraction/upload failures here",
    )
    p.add_argument(
        "--ocr-required-log", default="ocr_required.jsonl",
        help="append OCR-required (skipped) PDFs here",
    )
    p.add_argument(
        "--no-progress", action="store_true",
        help="suppress periodic progress lines (still prints summary)",
    )
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = {
        "pdf_bucket": args.s3_pdf_bucket
            or os.environ.get("NNM_S3_PDF_BUCKET")
            or os.environ.get("NNM_S3_BUCKET"),
        "pdf_prefix": args.s3_pdf_prefix
            or os.environ.get("NNM_S3_PDF_PREFIX")
            or os.environ.get("NNM_S3_PREFIX", ""),
        "ext_bucket": args.s3_extracted_bucket
            or os.environ.get("NNM_S3_EXTRACTED_BUCKET"),
        "ext_prefix": args.s3_extracted_prefix
            or os.environ.get("NNM_S3_EXTRACTED_PREFIX", "newnonmuncom-extracted/"),
        "region": args.region or os.environ.get("NNM_S3_REGION", "ap-northeast-2"),
        "csv_path": args.publication_csv or os.environ.get("NNM_PUBLICATION_CSV"),
        "csv_encoding": args.publication_csv_encoding
            or os.environ.get("NNM_PUBLICATION_CSV_ENCODING", "cp949"),
    }
    if not cfg["pdf_bucket"]:
        sys.exit("ERROR: NNM_S3_PDF_BUCKET or NNM_S3_BUCKET required "
                 "(or --s3-pdf-bucket)")
    if not cfg["ext_bucket"]:
        # extracted bucket 미지정 → PDF 버킷 fallback (다른 prefix 사용).
        cfg["ext_bucket"] = cfg["pdf_bucket"]
    return cfg


def load_legacy_aid_map(csv_path: str | None, encoding: str) -> dict[str, int]:
    """Returns {stem: legacy_aid}. CSV missing/unreadable → empty dict."""
    if not csv_path:
        return {}
    p = Path(csv_path)
    if not p.exists():
        print(f"WARNING: publication CSV not found at {csv_path}", file=sys.stderr)
        return {}
    result: dict[str, int] = {}
    with p.open("r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = (row.get("s3_key") or "").strip()
            raw = (row.get("legacy_aid") or "").strip()
            if not stem or not raw:
                continue
            try:
                result[stem] = int(raw)
            except ValueError:
                continue
    return result


def list_s3_keys(s3, bucket: str, prefix: str) -> Iterator[str]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def stem_of(s3_key: str) -> str:
    name = s3_key.rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


def index_extracted(s3, bucket: str, prefix: str) -> set[str]:
    """Stems with BOTH .json and .md present in the extracted prefix."""
    j, m = set(), set()
    for k in list_s3_keys(s3, bucket, prefix):
        name = k.rsplit("/", 1)[-1]
        if name.endswith(".json"):
            j.add(name[:-5])
        elif name.endswith(".md"):
            m.add(name[:-3])
    return j & m


def run_opendataloader(
    bin_path: str, pdf_path: Path, out_dir: Path, file_hash: str,
    *, threads: int, timeout_s: int,
) -> tuple[Path, Path]:
    """Run opendataloader-pdf (text-only, no hybrid/OCR). Returns (.json, .md)."""
    cmd = [
        bin_path,
        "-o", str(out_dir),
        "-f", "json,markdown",
        "--reading-order", "xycut",
        "--use-struct-tree",
        "--threads", str(threads),
        "--table-method", "cluster",
        "--image-output", "external",
        "--image-dir", str(out_dir / "images"),
        str(pdf_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"opendataloader-pdf exit {proc.returncode}: "
            f"{proc.stderr.decode(errors='ignore')[:500]}"
        )
    json_path = out_dir / f"{file_hash}.json"
    md_path = out_dir / f"{file_hash}.md"
    if not json_path.exists() or not md_path.exists():
        produced = sorted(p.name for p in out_dir.glob(f"{file_hash}.*"))
        raise RuntimeError(
            f"opendataloader output missing for {file_hash}; produced={produced}"
        )
    return json_path, md_path


def count_text_chars(raw: dict) -> int:
    """Recursively sum length of all `content` strings in the doc tree.

    Approximates server-side `_normalize_doc` + `_walk_kids` total chars.
    Used only to detect image-only / OCR-required PDFs.
    """
    total = 0

    def walk(node):
        nonlocal total
        if isinstance(node, dict):
            content = node.get("content")
            if isinstance(content, str):
                total += len(content)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw)
    return total


def append_jsonl(path: str | None, entry: dict) -> None:
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: failed to append {path}: {e}", file=sys.stderr)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)

    print(f"[config] pdf=s3://{cfg['pdf_bucket']}/{cfg['pdf_prefix']!r}")
    print(f"[config] extracted=s3://{cfg['ext_bucket']}/{cfg['ext_prefix']!r}")
    print(f"[config] region={cfg['region']} csv={cfg['csv_path']!r}")

    s3 = boto3.client("s3", region_name=cfg["region"])

    print(f"[1/3] indexing existing extracts ...", flush=True)
    t0 = time.time()
    done = index_extracted(s3, cfg["ext_bucket"], cfg["ext_prefix"])
    print(f"      done={len(done)} (both .json+.md) in {time.time()-t0:.1f}s")

    print(f"[2/3] listing PDFs ...", flush=True)
    t0 = time.time()
    pdfs = [
        k for k in list_s3_keys(s3, cfg["pdf_bucket"], cfg["pdf_prefix"])
        if k and not k.endswith("/")
    ]
    print(f"      total={len(pdfs)} in {time.time()-t0:.1f}s")

    legacy_map = load_legacy_aid_map(cfg["csv_path"], cfg["csv_encoding"])
    print(f"      csv entries (legacy_aid)={len(legacy_map)}")

    # legacy_aid DESC, CSV 미매핑은 뒤로.
    def sort_key(k: str) -> tuple[int, int, str]:
        s = stem_of(k)
        aid = legacy_map.get(s)
        return (1 if aid is None else 0, -(aid or 0), k)

    pdfs.sort(key=sort_key)

    print(f"[3/3] extracting (limit={args.limit}) ...", flush=True)
    ok = ocr_skip = failed = skipped_existing = 0
    started = time.time()

    for key in pdfs:
        processed = ok + ocr_skip + failed
        if args.limit is not None and processed >= args.limit:
            break

        stem = stem_of(key)
        if stem in done:
            skipped_existing += 1
            continue

        # 1. download PDF
        try:
            obj = s3.get_object(Bucket=cfg["pdf_bucket"], Key=key)
            data = obj["Body"].read()
        except ClientError as e:
            failed += 1
            append_jsonl(args.failed_log, {
                "s3_key": key, "stage": "download",
                "error": str(e), "failed_at": now_iso(),
            })
            continue

        file_hash = hashlib.sha256(data).hexdigest()
        size = len(data)

        # 2. opendataloader-pdf
        with tempfile.TemporaryDirectory() as td:
            pdf_tmp = Path(td) / f"{file_hash}.pdf"
            pdf_tmp.write_bytes(data)
            out_dir = Path(td) / "out"
            out_dir.mkdir()

            try:
                json_path, md_path = run_opendataloader(
                    args.opendataloader_bin, pdf_tmp, out_dir, file_hash,
                    threads=args.threads, timeout_s=args.extract_timeout_s,
                )
            except subprocess.TimeoutExpired:
                failed += 1
                append_jsonl(args.failed_log, {
                    "s3_key": key, "file_hash": file_hash, "size": size,
                    "stage": "extract", "error": "timeout",
                    "failed_at": now_iso(),
                })
                continue
            except RuntimeError as e:
                failed += 1
                append_jsonl(args.failed_log, {
                    "s3_key": key, "file_hash": file_hash, "size": size,
                    "stage": "extract", "error": str(e),
                    "failed_at": now_iso(),
                })
                continue

            json_bytes = json_path.read_bytes()
            md_bytes = md_path.read_bytes()

            # 3. parse + OCR-required check
            try:
                doc = json.loads(json_bytes)
            except json.JSONDecodeError as e:
                failed += 1
                append_jsonl(args.failed_log, {
                    "s3_key": key, "file_hash": file_hash, "size": size,
                    "stage": "parse", "error": str(e),
                    "failed_at": now_iso(),
                })
                continue

            text_chars = count_text_chars(doc)
            if text_chars < TEXT_MIN_CHARS:
                ocr_skip += 1
                append_jsonl(args.ocr_required_log, {
                    "s3_key": key, "file_hash": file_hash, "size": size,
                    "text_chars": text_chars,
                    "recorded_at": now_iso(),
                })
                continue

            # 4. inject file_hash + source_s3_key into JSON top-level.
            #    서버 측 embed-from-extracted 가 file_hash 로 dedup 하도록.
            doc["file_hash"] = file_hash
            doc["source_s3_key"] = key
            json_bytes_out = json.dumps(doc, ensure_ascii=False).encode("utf-8")

        # 5. upload (.json + .md) with file_hash in S3 object metadata too.
        ext_json_key = f"{cfg['ext_prefix']}{stem}.json"
        ext_md_key = f"{cfg['ext_prefix']}{stem}.md"
        try:
            s3.put_object(
                Bucket=cfg["ext_bucket"], Key=ext_json_key,
                Body=json_bytes_out,
                ContentType="application/json",
                Metadata={
                    "file-hash": file_hash,
                    "source-s3-key": key,
                },
            )
            s3.put_object(
                Bucket=cfg["ext_bucket"], Key=ext_md_key,
                Body=md_bytes,
                ContentType="text/markdown; charset=utf-8",
                Metadata={
                    "file-hash": file_hash,
                    "source-s3-key": key,
                },
            )
        except ClientError as e:
            failed += 1
            append_jsonl(args.failed_log, {
                "s3_key": key, "file_hash": file_hash, "size": size,
                "stage": "upload", "error": str(e),
                "failed_at": now_iso(),
            })
            continue

        ok += 1
        done.add(stem)

        if not args.no_progress and ok > 0 and ok % args.progress_every == 0:
            elapsed = time.time() - started
            rate = ok / max(elapsed, 1)
            attempted = ok + ocr_skip + failed
            remaining = (
                (args.limit - attempted) if args.limit is not None
                else (len(pdfs) - skipped_existing - attempted)
            )
            eta_s = remaining / max(rate, 0.001)
            print(
                f"  ok={ok} ocr_skip={ocr_skip} failed={failed} "
                f"existing_skipped={skipped_existing} "
                f"rate={rate:.2f}/s eta={eta_s/60:.1f}min",
                flush=True,
            )

    elapsed = time.time() - started
    print(f"\nDONE in {elapsed/60:.1f}min: ok={ok} ocr_skip={ocr_skip} "
          f"failed={failed} existing_skipped={skipped_existing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
