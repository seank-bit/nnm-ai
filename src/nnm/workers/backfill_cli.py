from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import structlog
import typer
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import AutoTokenizer

from nnm.config import get_settings
from nnm.db.session import get_factory
from nnm.errors import OcrRequiredError, PdfExtractionError
from nnm.infra.local_embedder import LocalEmbedder
from nnm.infra.repository import SqlBackfillRepository
from nnm.infra.s3 import S3Loader
from nnm.logging import configure_logging
from nnm.services.backfill import BackfillService
from nnm.services.chunker import Chunker
from nnm.services.maintenance import reset_database
from nnm.services.pdf_extractor import PdfExtractor
from nnm.services.publication_mapper import PublicationMapper
from nnm.services.title_filter import is_garbage_title

app = typer.Typer(help="nnm-ai 백필 CLI")
log = structlog.get_logger()


@app.callback()
def _setup() -> None:
    configure_logging()


@app.command()
def ingest(
    prefix: str = typer.Option("", help="S3 prefix"),
    limit: int | None = typer.Option(None, help="처리 개수 제한"),
    job_name: str = typer.Option("backfill", help="ingest_jobs.job_name"),
    skip_ocr: bool | None = typer.Option(
        None, "--skip-ocr/--no-skip-ocr",
        help="1차 텍스트 추출 elements=0 인 PDF (= OCR 필요) 를 건너뜀. "
             "기본값은 settings.pdf_skip_ocr.",
    ),
) -> None:
    """S3에서 PDF를 가져와 청킹·임베딩·적재."""
    asyncio.run(_run_ingest(
        prefix=prefix, limit=limit, job_name=job_name, skip_ocr=skip_ocr,
    ))


@app.command(name="reset-db")
def reset_db(force: bool = typer.Option(False, "--force")) -> None:
    """모든 nnm 테이블 drop 후 alembic upgrade head."""
    asyncio.run(reset_database(force=force))


@app.command(name="hnsw-disable")
def hnsw_disable() -> None:
    """대량 적재 전 HNSW 인덱스 drop."""
    asyncio.run(_run_hnsw(enable=False))


@app.command(name="hnsw-enable")
def hnsw_enable() -> None:
    """대량 적재 후 HNSW 인덱스 재생성."""
    asyncio.run(_run_hnsw(enable=True))


@app.command()
def reocr(
    limit: int | None = typer.Option(None, help="처리 개수 제한"),
    job_name: str = typer.Option("reocr", help="ingest_jobs.job_name"),
) -> None:
    """chunks=0 인 paper 를 S3 에서 다시 받아 OCR fallback 으로 재추출.

    1차 텍스트 추출에서 elements=0 이라 청크가 안 만들어진 paper 들을 대상으로
    삭제 후 재인제스트. PdfExtractor 가 자동으로 OCR fallback 실행.
    NNM_HYBRID_URL 미설정 시 동작 안 함 (warning).
    """
    asyncio.run(_run_reocr(limit=limit, job_name=job_name))


@app.command(name="embed-from-extracted")
def embed_from_extracted(
    since: str | None = typer.Option(
        None,
        help="LastModified >= 이 시각의 .json 만 처리. "
             "YYYY-MM-DD (UTC 자정) 또는 ISO-8601. "
             "미지정 시 날짜 필터 없음 (전체 .json 처리).",
    ),
    prefix: str | None = typer.Option(
        None,
        help="extracted prefix override. 기본: settings.s3_extracted_prefix.",
    ),
    limit: int | None = typer.Option(None, help="처리 개수 제한"),
    watch: bool = typer.Option(
        False, "--watch", help="True 면 한 라운드 후 종료하지 않고 주기 재스캔.",
    ),
    watch_interval_s: int = typer.Option(
        300, help="watch 모드 sleep 초 (기본 5분)",
    ),
    job_name: str = typer.Option(
        "embed-extracted", help="ingest_jobs.job_name",
    ),
    shard: str | None = typer.Option(
        None, "--shard",
        help="멀티 워커 분산. 형식 'i/N' (예: 0/3, 1/3, 2/3). "
             "각 워커는 crc32(stem) % N == i 인 키만 처리. None=모든 키.",
    ),
) -> None:
    """S3 extracted .json 을 읽어 chunks + embeddings 적재. PDF 미사용.

    .json 안의 file_hash (tools/local_extract.py 가 주입) 로 dedup.
    --watch 시 watch_interval_s 마다 재스캔하면서 신규 .json 만 추가 처리.
    --shard 로 여러 프로세스 분산 처리 가능 (각 프로세스 자체 GPU 모델 인스턴스).
    """
    shard_i, shard_n = _parse_shard(shard)
    asyncio.run(_run_embed_from_extracted(
        since=since, prefix=prefix, limit=limit,
        watch=watch, watch_interval_s=watch_interval_s, job_name=job_name,
        shard_i=shard_i, shard_n=shard_n,
    ))


def _parse_shard(shard: str | None) -> tuple[int | None, int | None]:
    if not shard:
        return None, None
    try:
        i_str, n_str = shard.split("/", 1)
        i, n = int(i_str), int(n_str)
        if not (0 <= i < n) or n <= 0:
            raise ValueError
        return i, n
    except Exception:
        raise typer.BadParameter(f"--shard 는 'i/N' 형식이어야 함 (예: 0/3). got={shard!r}")


@app.command(name="extract-only")
def extract_only(
    prefix: str = typer.Option("", help="S3 prefix (PDF 버킷). 미지정 시 settings 사용."),
    limit: int | None = typer.Option(None, help="처리 개수 제한 (기존 skip 제외)"),
    progress_every: int = typer.Option(50, help="진행 로그 주기"),
) -> None:
    """이미 추출 산출물(.json+.md)이 있는 PDF 는 skip, 나머지만 추출 & 업로드.

    DB 는 건드리지 않음. OCR 필요한 PDF (1차 텍스트 부실) 는 자동 skip
    (skip_ocr=True 강제). 추출 실패는 failed_pdfs.jsonl 에 기록.
    """
    asyncio.run(_run_extract_only(
        prefix=prefix, limit=limit, progress_every=progress_every,
    ))


@app.command(name="clear-ocr")
def clear_ocr(
    yes: bool = typer.Option(False, "--yes", "-y", help="확인 프롬프트 생략"),
    dry_run: bool = typer.Option(False, "--dry-run", help="삭제 없이 대상만 출력"),
) -> None:
    """과거 reocr 작업으로 등록된 paper 와 그 ingest_job 을 모두 삭제.

    대상: ingest_jobs.job_name LIKE 'reocr%' 인 job 들의 paper_id.
    chunks / chunk_embeddings / paper_references 는 FK CASCADE 로 자동 삭제.
    """
    asyncio.run(_run_clear_ocr(yes=yes, dry_run=dry_run))


@app.command()
def remap(
    all_rows: bool = typer.Option(
        False, "--all",
        help="external_id 가 이미 있는 paper 도 재매핑 (기본: NULL 인 것만).",
    ),
    overwrite_title: bool = typer.Option(
        False, "--overwrite-title",
        help="publications CSV 의 title 로 paper.title 을 무조건 덮어쓰기.",
    ),
    limit: int | None = typer.Option(None, help="처리 개수 제한"),
) -> None:
    """기존 papers 의 external_id / title 을 publications CSV 매핑으로 채움."""
    asyncio.run(_run_remap(all_rows=all_rows, overwrite_title=overwrite_title, limit=limit))


async def _create_job(session: AsyncSession, *, job_name: str, prefix: str) -> int:
    result = await session.execute(
        text("INSERT INTO ingest_jobs(job_name, s3_prefix) VALUES (:n, :p) RETURNING id"),
        {"n": job_name, "p": prefix},
    )
    job_id = result.scalar_one()
    await session.commit()
    return job_id


async def _enqueue_item(session: AsyncSession, job_id: int, s3_key: str) -> None:
    await session.execute(
        text(
            "INSERT INTO ingest_job_items(job_id, s3_key) VALUES (:j, :k) "
            "ON CONFLICT (job_id, s3_key) DO NOTHING"
        ),
        {"j": job_id, "k": s3_key},
    )
    await session.commit()


async def _finalize_job(session: AsyncSession, job_id: int, *, processed: int) -> None:
    await session.execute(
        text("UPDATE ingest_jobs SET processed = :p, finished_at = now() WHERE id = :id"),
        {"p": processed, "id": job_id},
    )
    await session.commit()


def _resolve_failed_record_path(settings) -> Path | None:
    """settings.failed_pdfs_path 를 storage_root 기준 절대경로로 변환.

    None / 빈 문자열이면 None 반환 (기록 비활성).
    """
    p = settings.failed_pdfs_path
    if p is None:
        return None
    p = Path(p)
    if p.is_absolute():
        return p
    return Path(settings.storage_root) / p


def _build_extracted_uploader(settings) -> tuple[S3Loader | None, str]:
    """추출 결과 .json/.md 를 보관할 S3 클라이언트와 prefix.

    bucket 미설정 시 PDF 버킷 → 기본 버킷 순으로 fallback.
    """
    bucket = (
        settings.s3_extracted_bucket
        or settings.s3_pdf_bucket
        or settings.s3_bucket
    )
    if not bucket:
        return None, ""
    return (
        S3Loader(bucket=bucket, region=settings.s3_region),
        settings.s3_extracted_prefix,
    )


async def _run_ingest(
    *, prefix: str, limit: int | None, job_name: str, skip_ocr: bool | None = None,
) -> None:
    settings = get_settings()
    storage_root = Path(settings.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    effective_skip_ocr = (
        settings.pdf_skip_ocr if skip_ocr is None else skip_ocr
    )
    s3_bucket = settings.s3_pdf_bucket or settings.s3_bucket
    s3 = S3Loader(bucket=s3_bucket, region=settings.s3_region)
    extractor = PdfExtractor(
        threads=settings.pdf_threads,
        hybrid_url=settings.hybrid_url,
        hybrid_mode=settings.hybrid_mode,
        hybrid_timeout_ms=settings.hybrid_timeout_ms,
        extract_timeout_s=settings.pdf_extract_timeout_s,
        skip_ocr=effective_skip_ocr,
    )
    log.info("backfill.skip_ocr", enabled=effective_skip_ocr)
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    chunker = Chunker(
        tokenizer=tokenizer,
        target_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    embedder = LocalEmbedder(settings=settings)
    embedder.load()

    factory = get_factory()
    processed = 0
    mapper = PublicationMapper(
        csv_path=settings.publication_csv,
        encoding=settings.publication_csv_encoding,
    )
    extracted_uploader, extracted_prefix = _build_extracted_uploader(settings)
    failed_record_path = _resolve_failed_record_path(settings)
    async with factory() as session:
        repo = SqlBackfillRepository(db=session)
        svc = BackfillService(
            s3=s3, repo=repo, mapper=mapper,
            extractor=extractor, chunker=chunker, embedder=embedder,
            storage_root=storage_root,
            extracted_uploader=extracted_uploader,
            extracted_prefix=extracted_prefix,
            failed_record_path=failed_record_path,
        )

        job_id = await _create_job(session, job_name=job_name, prefix=prefix)

        effective_prefix = prefix or settings.s3_pdf_prefix or settings.s3_prefix

        # CSV legacy_aid DESC 우선 처리. CSV 미매핑 키는 뒤로 밀린다.
        keys: list[str] = []
        async for key in s3.list_keys(effective_prefix):
            if not key or key.endswith("/"):
                continue
            keys.append(key)

        def _sort_key(k: str) -> tuple[int, int, str]:
            aid = mapper.legacy_aid_for(k)
            # (매핑 없음 그룹, -legacy_aid, key) → 매핑 있는 것이 먼저, 그 안에서 DESC
            return (1 if aid is None else 0, -(aid or 0), k)

        keys.sort(key=_sort_key)
        log.info("backfill.order", total=len(keys), order="legacy_aid_desc")

        for key in keys:
            await _enqueue_item(session, job_id, key)
            result = await svc.process_one(job_id=job_id, s3_key=key)
            log.info(
                "backfill.iter",
                key=key,
                legacy_aid=mapper.legacy_aid_for(key),
                result=result,
            )
            processed += 1
            if limit is not None and processed >= limit:
                break

        await _finalize_job(session, job_id, processed=processed)


def _parse_since(since: str | None) -> datetime | None:
    """YYYY-MM-DD 또는 ISO-8601 → UTC datetime. None/빈문자열 → None (필터 미적용)."""
    if not since or not since.strip():
        return None
    s = since.strip()
    if len(s) == 10 and s.count("-") == 2:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


async def _run_embed_from_extracted(
    *, since: str | None, prefix: str | None, limit: int | None,
    watch: bool, watch_interval_s: int, job_name: str,
    shard_i: int | None, shard_n: int | None,
) -> None:
    import zlib
    settings = get_settings()
    storage_root = Path(settings.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    since_dt = _parse_since(since)
    log.info(
        "embed_extracted.config",
        since=since_dt.isoformat() if since_dt else None,
        watch=watch, interval_s=watch_interval_s,
    )
    typer.echo(
        f"since = {since_dt.isoformat() if since_dt else '(none — process all)'}"
        f"  watch={watch}"
    )

    # extracted bucket 클라이언트가 우선 필요 (download + list)
    uploader, default_prefix = _build_extracted_uploader(settings)
    if uploader is None:
        typer.echo("ERROR: extracted bucket/prefix 미설정 (NNM_S3_EXTRACTED_*).")
        raise typer.Exit(code=1)
    ext_prefix = prefix or default_prefix
    typer.echo(f"extracted = s3://{uploader.bucket}/{ext_prefix!r}")

    # 일반 backfill 파이프라인 wiring (PdfExtractor 는 호출 안 되지만 필드 필요)
    s3_bucket = settings.s3_pdf_bucket or settings.s3_bucket
    s3 = S3Loader(bucket=s3_bucket, region=settings.s3_region)
    extractor = PdfExtractor(
        threads=settings.pdf_threads,
        hybrid_url=settings.hybrid_url,
        hybrid_mode=settings.hybrid_mode,
        hybrid_timeout_ms=settings.hybrid_timeout_ms,
        extract_timeout_s=settings.pdf_extract_timeout_s,
        skip_ocr=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    chunker = Chunker(
        tokenizer=tokenizer,
        target_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    embedder = LocalEmbedder(settings=settings)
    embedder.load()
    mapper = PublicationMapper(
        csv_path=settings.publication_csv,
        encoding=settings.publication_csv_encoding,
    )
    failed_record_path = _resolve_failed_record_path(settings)

    pdf_prefix = settings.s3_pdf_prefix or settings.s3_prefix or ""

    factory = get_factory()
    processed_total = 0
    seen_keys: set[str] = set()  # watch 라운드 간 중복 처리 방지

    round_idx = 0
    while True:
        round_idx += 1
        async with factory() as session:
            repo = SqlBackfillRepository(db=session)
            svc = BackfillService(
                s3=s3, repo=repo, mapper=mapper,
                extractor=extractor, chunker=chunker, embedder=embedder,
                storage_root=storage_root,
                extracted_uploader=uploader,
                extracted_prefix=ext_prefix,
                failed_record_path=failed_record_path,
                pdf_prefix=pdf_prefix,
            )
            shard_suffix = (
                f"-s{shard_i}" if shard_n is not None else ""
            )
            job_id = await _create_job(
                session,
                job_name=f"{job_name}{shard_suffix}-r{round_idx}",
                prefix=ext_prefix,
            )

            # scan extracted prefix → (since 가 있으면 LastModified 필터) → .json 만
            # → (shard 가 있으면 crc32(stem) % N == i 만)
            scanned = 0
            candidates: list[str] = []
            async for key, meta in uploader.list_keys_with_meta(ext_prefix):
                scanned += 1
                if not key.endswith(".json"):
                    continue
                if since_dt is not None:
                    lm = meta.get("last_modified")
                    if lm is None or lm < since_dt:
                        continue
                if shard_n is not None and shard_i is not None:
                    name = key.rsplit("/", 1)[-1]
                    stem = name.rsplit(".", 1)[0]
                    if zlib.crc32(stem.encode()) % shard_n != shard_i:
                        continue
                if key in seen_keys:
                    continue
                candidates.append(key)

            since_label = (
                f"since={since_dt.date()}" if since_dt is not None else "all"
            )
            shard_label = (
                f" shard={shard_i}/{shard_n}"
                if shard_n is not None else ""
            )
            typer.echo(
                f"[round {round_idx}{shard_label}] scanned={scanned} "
                f"candidates(.json {since_label})={len(candidates)}"
            )

            ok = skipped = failed_cnt = 0
            for key in candidates:
                if limit is not None and processed_total >= limit:
                    break
                seen_keys.add(key)
                await _enqueue_item(session, job_id, key)
                try:
                    result = await svc.process_extracted_one(
                        job_id=job_id, extracted_json_key=key,
                    )
                except Exception as e:  # noqa: BLE001
                    log.error(
                        "embed_extracted.unhandled",
                        key=key, error=str(e),
                    )
                    failed_cnt += 1
                    processed_total += 1
                    continue
                if result == "ok":
                    ok += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    failed_cnt += 1
                processed_total += 1

            await _finalize_job(session, job_id, processed=ok + skipped + failed_cnt)
            typer.echo(
                f"[round {round_idx}] ok={ok} skipped={skipped} "
                f"failed={failed_cnt} (total processed={processed_total})"
            )

        if limit is not None and processed_total >= limit:
            typer.echo(f"limit={limit} 도달 → 종료")
            break
        if not watch:
            break
        typer.echo(f"watch sleep {watch_interval_s}s ...")
        await asyncio.sleep(watch_interval_s)

    log.info(
        "embed_extracted.done",
        rounds=round_idx, processed_total=processed_total,
    )


async def _run_extract_only(
    *, prefix: str, limit: int | None, progress_every: int,
) -> None:
    """PDF 별 .json/.md 산출물을 S3 에 일괄 생성. DB 미사용.

    1) 기존 extracted prefix 인덱싱 → stem set (json AND md 둘 다 있어야 done)
    2) PDF list (legacy_aid DESC)
    3) stem 미존재 → download → extract(skip_ocr=True) → upload
       - OcrRequiredError: skip + count
       - PdfExtractionError: failed_pdfs.jsonl 기록 + skip
    """
    settings = get_settings()
    storage_root = Path(settings.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    s3_bucket = settings.s3_pdf_bucket or settings.s3_bucket
    s3 = S3Loader(bucket=s3_bucket, region=settings.s3_region)

    extractor = PdfExtractor(
        threads=settings.pdf_threads,
        hybrid_url=settings.hybrid_url,
        hybrid_mode=settings.hybrid_mode,
        hybrid_timeout_ms=settings.hybrid_timeout_ms,
        extract_timeout_s=settings.pdf_extract_timeout_s,
        skip_ocr=True,  # extract-only 는 항상 OCR skip
    )
    uploader, ext_prefix = _build_extracted_uploader(settings)
    if uploader is None:
        typer.echo("ERROR: extracted bucket/prefix 미설정 (NNM_S3_EXTRACTED_*).")
        raise typer.Exit(code=1)

    mapper = PublicationMapper(
        csv_path=settings.publication_csv,
        encoding=settings.publication_csv_encoding,
    )
    failed_record_path = _resolve_failed_record_path(settings)

    async def _record_fail(s3_key: str, file_hash: str, size: int, err: Exception) -> None:
        if failed_record_path is None:
            return
        entry = {
            "s3_key": s3_key,
            "file_hash": file_hash,
            "file_size_bytes": size,
            "error_type": type(err).__name__,
            "error": str(err),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            failed_record_path.parent.mkdir(parents=True, exist_ok=True)
            with failed_record_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    # 1) 기존 산출물 인덱싱
    typer.echo(f"기존 산출물 인덱싱 중... bucket={uploader.bucket} prefix={ext_prefix!r}")
    json_stems: set[str] = set()
    md_stems: set[str] = set()
    async for k in uploader.list_keys(ext_prefix):
        if not k or k.endswith("/"):
            continue
        name = k.rsplit("/", 1)[-1]
        if name.endswith(".json"):
            json_stems.add(name[:-5])
        elif name.endswith(".md"):
            md_stems.add(name[:-3])
    done_stems = json_stems & md_stems
    typer.echo(
        f"기존: json={len(json_stems)} md={len(md_stems)} "
        f"both={len(done_stems)}"
    )

    # 2) PDF list (legacy_aid DESC)
    effective_prefix = prefix or settings.s3_pdf_prefix or settings.s3_prefix
    typer.echo(f"PDF 목록 수집 중... bucket={s3_bucket} prefix={effective_prefix!r}")
    pdf_keys: list[str] = []
    async for key in s3.list_keys(effective_prefix):
        if not key or key.endswith("/"):
            continue
        pdf_keys.append(key)
    typer.echo(f"PDF 총 {len(pdf_keys)} 건")

    def _stem_of(k: str) -> str:
        name = k.rsplit("/", 1)[-1]
        if "." in name:
            name = name.rsplit(".", 1)[0]
        return name

    def _sort_key(k: str) -> tuple[int, int, str]:
        aid = mapper.legacy_aid_for(k)
        return (1 if aid is None else 0, -(aid or 0), k)

    pdf_keys.sort(key=_sort_key)

    # 3) Walk
    skipped_existing = 0
    extracted_ok = 0
    skipped_ocr = 0
    failed_count = 0

    for key in pdf_keys:
        # limit 은 "실제 시도(processed)" 수 기준
        processed = extracted_ok + skipped_ocr + failed_count
        if limit is not None and processed >= limit:
            break
        stem = _stem_of(key)
        if stem in done_stems:
            skipped_existing += 1
            continue

        try:
            data, digest = await s3.download(key)
        except Exception as e:  # noqa: BLE001 — S3 일시 오류 → 건너뜀
            log.warning("extract_only.download_failed", s3_key=key, error=str(e))
            failed_count += 1
            continue

        try:
            extraction = await extractor.extract(data, file_hash=digest)
        except OcrRequiredError:
            log.info("extract_only.ocr_skip", s3_key=key)
            skipped_ocr += 1
            continue
        except PdfExtractionError as e:
            log.warning("extract_only.extract_failed", s3_key=key, error=str(e))
            await _record_fail(key, digest, len(data), e)
            failed_count += 1
            continue

        json_key = f"{ext_prefix}{stem}.json"
        md_key = f"{ext_prefix}{stem}.md"
        try:
            await uploader.upload_bytes(
                json_key, extraction.json_bytes, content_type="application/json",
            )
            await uploader.upload_bytes(
                md_key, extraction.md_bytes,
                content_type="text/markdown; charset=utf-8",
            )
        except Exception as e:  # noqa: BLE001 — 업로드 일시 실패
            log.warning("extract_only.upload_failed", s3_key=key, error=str(e))
            failed_count += 1
            continue

        extracted_ok += 1
        done_stems.add(stem)  # 같은 stem 중복 처리 방지 (혹시 모를 dedup)
        total = extracted_ok + skipped_ocr + failed_count
        if extracted_ok % progress_every == 0:
            typer.echo(
                f"진행 {total}: ok={extracted_ok} ocr_skip={skipped_ocr} "
                f"failed={failed_count} existing_skipped={skipped_existing}"
            )
            log.info(
                "extract_only.progress",
                ok=extracted_ok, ocr_skip=skipped_ocr,
                failed=failed_count, existing=skipped_existing,
            )

    log.info(
        "extract_only.done",
        ok=extracted_ok, ocr_skip=skipped_ocr,
        failed=failed_count, existing=skipped_existing,
    )
    typer.echo(
        f"완료: 신규={extracted_ok} ocr_skip={skipped_ocr} "
        f"failed={failed_count} 기존skip={skipped_existing}"
    )


async def _run_reocr(*, limit: int | None, job_name: str) -> None:
    """chunks=0 paper 를 삭제하고 같은 s3_key 로 재인제스트.

    재추출 시 PdfExtractor 가 1차 텍스트 → 0이면 OCR fallback 자동 수행.
    """
    settings = get_settings()
    storage_root = Path(settings.storage_root)

    if not settings.hybrid_url:
        log.warning(
            "reocr.no_hybrid",
            message="NNM_HYBRID_URL 미설정. OCR fallback 없이는 재처리 효과 없음.",
        )

    factory = get_factory()
    async with factory() as session:
        # chunks=0 인 paper 의 s3_key 목록 + 삭제 대상 id
        rows = (await session.execute(text(
            "SELECT p.id, p.s3_key FROM papers p "
            "WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.paper_id = p.id) "
            "ORDER BY p.id"
        ))).all()
        targets = rows if limit is None else rows[:limit]

    if not targets:
        log.info("reocr.no_targets")
        typer.echo("chunks=0 paper 없음. 작업 종료.")
        return

    log.info("reocr.start", count=len(targets))
    typer.echo(f"재처리 대상: {len(targets)} 건")

    # ingest 파이프라인과 동일한 구성 (재사용)
    s3_bucket = settings.s3_pdf_bucket or settings.s3_bucket
    s3 = S3Loader(bucket=s3_bucket, region=settings.s3_region)
    extractor = PdfExtractor(
        threads=settings.pdf_threads,
        hybrid_url=settings.hybrid_url,
        hybrid_mode=settings.hybrid_mode,
        hybrid_timeout_ms=settings.hybrid_timeout_ms,
        extract_timeout_s=settings.pdf_extract_timeout_s,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    chunker = Chunker(
        tokenizer=tokenizer,
        target_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    embedder = LocalEmbedder(settings=settings)
    embedder.load()
    mapper = PublicationMapper(
        csv_path=settings.publication_csv,
        encoding=settings.publication_csv_encoding,
    )

    extracted_uploader, extracted_prefix = _build_extracted_uploader(settings)
    failed_record_path = _resolve_failed_record_path(settings)
    ok = failed = 0
    async with factory() as session:
        repo = SqlBackfillRepository(db=session)
        svc = BackfillService(
            s3=s3, repo=repo, mapper=mapper,
            extractor=extractor, chunker=chunker, embedder=embedder,
            storage_root=storage_root,
            extracted_uploader=extracted_uploader,
            extracted_prefix=extracted_prefix,
            failed_record_path=failed_record_path,
        )
        job_id = await _create_job(session, job_name=job_name, prefix="(reocr)")

        for r in targets:
            # 기존 paper + 관련 ingest_job_items 삭제 (file_hash dedup 해제)
            await session.execute(
                text("DELETE FROM ingest_job_items WHERE paper_id = :id"),
                {"id": r.id},
            )
            await session.execute(
                text("DELETE FROM papers WHERE id = :id"), {"id": r.id},
            )
            await session.commit()

            await _enqueue_item(session, job_id, r.s3_key)
            result = await svc.process_one(job_id=job_id, s3_key=r.s3_key)
            log.info("reocr.iter", s3_key=r.s3_key, old_id=r.id, result=result)
            if result == "ok":
                ok += 1
            else:
                failed += 1

        await _finalize_job(session, job_id, processed=len(targets))

    log.info("reocr.done", ok=ok, failed=failed, total=len(targets))
    typer.echo(f"재처리 완료: ok={ok} failed={failed}")


async def _run_clear_ocr(*, yes: bool, dry_run: bool) -> None:
    factory = get_factory()
    async with factory() as session:
        # reocr 류 job 식별. 사용자가 지정한 job_name 변형까지 잡으려고 LIKE 사용.
        jobs = (await session.execute(text(
            "SELECT id, job_name FROM ingest_jobs WHERE job_name LIKE 'reocr%'"
        ))).all()
        if not jobs:
            typer.echo("reocr 류 ingest_jobs 없음. 종료.")
            return

        job_ids = [j.id for j in jobs]
        paper_rows = (await session.execute(
            text(
                "SELECT DISTINCT paper_id FROM ingest_job_items "
                "WHERE job_id IN :job_ids AND paper_id IS NOT NULL"
            ).bindparams(bindparam("job_ids", expanding=True)),
            {"job_ids": job_ids},
        )).all()
        paper_ids = [r.paper_id for r in paper_rows]

        typer.echo(f"대상: jobs={len(jobs)} papers={len(paper_ids)}")
        for j in jobs:
            typer.echo(f"  job_id={j.id} name={j.job_name}")

        if dry_run:
            typer.echo("--dry-run: 변경 없이 종료.")
            return

        if not yes:
            confirm = typer.confirm(
                f"papers {len(paper_ids)} 건 + reocr jobs {len(jobs)} 건 삭제할까요?"
            )
            if not confirm:
                typer.echo("취소됨.")
                return

        if paper_ids:
            # papers FK CASCADE 로 chunks / chunk_embeddings / paper_references 동반 삭제.
            # ingest_job_items.paper_id 는 ON DELETE 미설정 → 먼저 unset.
            await session.execute(
                text(
                    "UPDATE ingest_job_items SET paper_id = NULL "
                    "WHERE paper_id IN :ids"
                ).bindparams(bindparam("ids", expanding=True)),
                {"ids": paper_ids},
            )
            await session.execute(
                text("DELETE FROM papers WHERE id IN :ids")
                .bindparams(bindparam("ids", expanding=True)),
                {"ids": paper_ids},
            )
        # reocr job 자체도 정리 (items 는 CASCADE 로 함께 삭제).
        await session.execute(
            text("DELETE FROM ingest_jobs WHERE id IN :ids")
            .bindparams(bindparam("ids", expanding=True)),
            {"ids": job_ids},
        )
        await session.commit()

    log.info(
        "clear_ocr.done",
        papers_deleted=len(paper_ids), jobs_deleted=len(jobs),
    )
    typer.echo(
        f"삭제 완료: papers={len(paper_ids)} jobs={len(jobs)}"
    )


async def _run_hnsw(*, enable: bool) -> None:
    factory = get_factory()
    async with factory() as session:
        if enable:
            log.info("hnsw.rebuild.start")
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS chunk_emb_dense_hnsw "
                "ON chunk_embeddings USING hnsw (dense vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            ))
            await session.commit()
            log.info("hnsw.rebuild.done")
            typer.echo("HNSW 인덱스 재생성 완료")
        else:
            log.info("hnsw.drop.start")
            await session.execute(text("DROP INDEX IF EXISTS chunk_emb_dense_hnsw"))
            await session.commit()
            log.info("hnsw.drop.done")
            typer.echo("HNSW 인덱스 drop 완료 (대량 적재 후 nnm hnsw-enable 로 복구)")


async def _run_remap(*, all_rows: bool, overwrite_title: bool, limit: int | None) -> None:
    settings = get_settings()
    factory = get_factory()

    scanned = 0
    matched = 0
    updated = 0

    mapper = PublicationMapper(
        csv_path=settings.publication_csv,
        encoding=settings.publication_csv_encoding,
    )
    if settings.publication_csv is None:
        log.error("remap.no_csv", message="NNM_PUBLICATION_CSV 가 .env 에 없음")
        return

    async with factory() as session:
        where = "" if all_rows else "WHERE external_id IS NULL"
        sql = f"SELECT id, s3_key, external_id, title FROM papers {where} ORDER BY id"
        rows = (await session.execute(text(sql))).all()
        targets = rows if limit is None else rows[:limit]

        for r in targets:
            scanned += 1
            mapping = await mapper.lookup(r.s3_key)
            if mapping is None:
                continue
            matched += 1
            existing_ok = bool(r.title) and not is_garbage_title(r.title)
            if mapping.title and (overwrite_title or not existing_ok):
                new_title = mapping.title
            elif existing_ok:
                new_title = r.title
            else:
                new_title = None
            await session.execute(
                text(
                    "UPDATE papers SET external_id = :eid, title = :title "
                    "WHERE id = :id"
                ),
                {"eid": str(mapping.publication_id), "title": new_title, "id": r.id},
            )
            updated += 1
            if updated % 100 == 0:
                await session.commit()
                log.info("remap.progress", scanned=scanned, matched=matched, updated=updated)
        await session.commit()

    log.info("remap.done", scanned=scanned, matched=matched, updated=updated)
    typer.echo(f"remap 완료: scanned={scanned} matched={matched} updated={updated}")


if __name__ == "__main__":
    app()
