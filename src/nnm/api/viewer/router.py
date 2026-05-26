from __future__ import annotations
import datetime as dt
import json
from pathlib import Path

import structlog
from fastapi import APIRouter, Form, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, func, select

from nnm.api.deps import DbDep, EmbedderDep
from nnm.config import get_settings
from nnm.db.models import Chunk, ChunkEmbedding, IngestJob, IngestJobItem, Paper
from nnm.errors import PaperNotFound
from nnm.infra.groq_client import GroqError
from nnm.infra.s3 import S3Loader
from nnm.services import rag as rag_service

router = APIRouter()
_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))
log = structlog.get_logger()

PAGE_SIZE = 50
SPARK_DAYS = 7
S3_COUNT_TTL_SECONDS = 24 * 3600
S3_COUNT_CACHE_FILE = ".s3_count.json"


def _read_s3_cache(path: Path) -> tuple[int, dt.datetime] | None:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data["count"]), dt.datetime.fromisoformat(data["fetched_at"])
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard.s3_cache_read_failed", path=str(path), error=str(exc))
        return None


def _write_s3_cache(path: Path, count: int, fetched_at: dt.datetime) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"count": count, "fetched_at": fetched_at.isoformat()}),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard.s3_cache_write_failed", path=str(path), error=str(exc))


async def _get_s3_object_count() -> tuple[int | None, dt.datetime | None]:
    s = get_settings()
    bucket = s.s3_pdf_bucket or s.s3_bucket
    prefix = s.s3_pdf_prefix or s.s3_prefix
    if not bucket:
        return None, None
    cache_path = Path(s.storage_root) / S3_COUNT_CACHE_FILE
    now = dt.datetime.now(dt.timezone.utc)
    cached = _read_s3_cache(cache_path)
    if cached and (now - cached[1]).total_seconds() < S3_COUNT_TTL_SECONDS:
        return cached[0], cached[1]
    try:
        loader = S3Loader(bucket=bucket, region=s.s3_region)
        count = await loader.count_objects(prefix)
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard.s3_count_failed", bucket=bucket, error=str(exc))
        if cached:
            return cached[0], cached[1]
        return None, None
    _write_s3_cache(cache_path, count, now)
    return count, now


def _bucket_by_day(rows, days):
    today = dt.datetime.now(dt.timezone.utc).date()
    series = {today - dt.timedelta(days=days - 1 - i): 0 for i in range(days)}
    for d, cnt in rows:
        key = d.date() if isinstance(d, dt.datetime) else d
        if key in series:
            series[key] = int(cnt)
    return [series[k] for k in sorted(series.keys())]


@router.get("/")
async def dashboard(request: Request, db: DbDep):
    total_papers = (await db.execute(select(func.count(Paper.id)))).scalar_one()
    total_chunks = (await db.execute(select(func.count(Chunk.id)))).scalar_one()
    by_status_rows = (await db.execute(
        select(Paper.status, func.count(Paper.id)).group_by(Paper.status)
    )).all()
    by_status = {row[0]: row[1] for row in by_status_rows}

    mapped = (await db.execute(
        select(func.count(Paper.id)).where(Paper.external_id.isnot(None))
    )).scalar_one()
    mapping_rate = round(100 * mapped / total_papers, 1) if total_papers else 0.0

    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=SPARK_DAYS)
    paper_day = func.date_trunc("day", Paper.created_at).label("day")
    emb_day = func.date_trunc("day", ChunkEmbedding.created_at).label("day")
    papers_day_rows = (await db.execute(
        select(paper_day, func.count(Paper.id))
        .where(Paper.created_at >= since)
        .group_by(paper_day)
    )).all()
    chunks_day_rows = (await db.execute(
        select(emb_day, func.count(ChunkEmbedding.chunk_id))
        .where(ChunkEmbedding.created_at >= since)
        .group_by(emb_day)
    )).all()
    mapped_day_rows = (await db.execute(
        select(
            paper_day,
            func.count(case((Paper.external_id.isnot(None), Paper.id))),
            func.count(Paper.id),
        )
        .where(Paper.created_at >= since)
        .group_by(paper_day)
    )).all()

    papers_spark = _bucket_by_day(papers_day_rows, SPARK_DAYS)
    chunks_spark = _bucket_by_day(chunks_day_rows, SPARK_DAYS)

    today = dt.datetime.now(dt.timezone.utc).date()
    mapped_series = {
        today - dt.timedelta(days=SPARK_DAYS - 1 - i): (0, 0) for i in range(SPARK_DAYS)
    }
    for d, mapped_cnt, total_cnt in mapped_day_rows:
        key = d.date() if isinstance(d, dt.datetime) else d
        if key in mapped_series:
            mapped_series[key] = (int(mapped_cnt), int(total_cnt))
    mapping_spark = [
        round(100 * m / t, 1) if t else 0.0
        for m, t in (mapped_series[k] for k in sorted(mapped_series.keys()))
    ]

    papers_today_delta = papers_spark[-1] if papers_spark else 0
    chunks_today_delta = chunks_spark[-1] if chunks_spark else 0
    mapping_delta_pp = (
        round(mapping_spark[-1] - mapping_spark[0], 1)
        if len(mapping_spark) >= 2 else 0.0
    )

    recent_jobs = (await db.scalars(
        select(IngestJob).order_by(IngestJob.started_at.desc()).limit(5)
    )).all()

    s3_object_count, s3_count_fetched_at = await _get_s3_object_count()

    return templates.TemplateResponse(
        request, "dashboard.html",
        {
            "total_papers": total_papers, "total_chunks": total_chunks,
            "by_status": by_status, "mapping_rate": mapping_rate,
            "recent_jobs": recent_jobs,
            "papers_spark": papers_spark,
            "chunks_spark": chunks_spark,
            "mapping_spark": mapping_spark,
            "papers_today_delta": papers_today_delta,
            "chunks_today_delta": chunks_today_delta,
            "mapping_delta_pp": mapping_delta_pp,
            "s3_object_count": s3_object_count,
            "s3_count_fetched_at": s3_count_fetched_at,
        },
    )


@router.get("/papers")
async def papers_list(
    request: Request, db: DbDep,
    page: int = 1, language: str | None = None, status: str | None = None,
):
    q = select(
        Paper.id, Paper.title, Paper.external_id, Paper.language,
        Paper.published_year, Paper.status,
        select(func.count(Chunk.id))
        .where(Chunk.paper_id == Paper.id).scalar_subquery().label("chunk_count"),
    )
    if language:
        q = q.where(Paper.language == language)
    if status:
        q = q.where(Paper.status == status)
    q = q.order_by(Paper.id.desc()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE + 1)

    rows = (await db.execute(q)).all()
    has_next = len(rows) > PAGE_SIZE
    papers = rows[:PAGE_SIZE]
    return templates.TemplateResponse(
        request, "papers_list.html",
        {
            "papers": papers, "page": page, "has_next": has_next,
            "filters": {"language": language, "status": status},
        },
    )


@router.get("/papers/{paper_id}")
async def paper_detail(request: Request, paper_id: int, db: DbDep):
    paper = await db.get(Paper, paper_id)
    if paper is None:
        raise PaperNotFound(f"paper {paper_id} not found")
    chunks = (await db.scalars(
        select(Chunk).where(Chunk.paper_id == paper_id).order_by(Chunk.seq)
    )).all()
    return templates.TemplateResponse(
        request, "paper_detail.html",
        {"paper": paper, "chunks": chunks},
    )


@router.get("/chunks/{chunk_id}")
async def chunk_detail(request: Request, chunk_id: int, db: DbDep):
    chunk = await db.get(Chunk, chunk_id)
    if chunk is None:
        raise PaperNotFound(f"chunk {chunk_id} not found")
    embedding = (await db.scalars(
        select(ChunkEmbedding).where(ChunkEmbedding.chunk_id == chunk_id)
    )).first()
    return templates.TemplateResponse(
        request, "chunk_detail.html",
        {"chunk": chunk, "embedding": embedding},
    )


@router.get("/rag")
async def rag_form(request: Request):
    s = get_settings()
    return templates.TemplateResponse(
        request, "rag.html",
        {
            "question": "",
            "result": None,
            "error": None,
            "model": s.groq_model,
            "key_configured": bool(s.groq_api_key),
            "top_k": s.rag_top_k,
        },
    )


@router.post("/rag")
async def rag_submit(
    request: Request,
    db: DbDep,
    embedder: EmbedderDep,
    question: str = Form(...),
):
    s = get_settings()
    error: str | None = None
    result = None
    q = (question or "").strip()
    if not q:
        error = "질문을 입력하세요."
    elif not s.groq_api_key:
        error = "NNM_GROQ_API_KEY 가 비어 있습니다. .env 에 키를 채워주세요."
    else:
        try:
            result = await rag_service.answer(db, embedder, s, q)
        except GroqError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("rag.failed", error=str(exc))
            error = f"RAG 실행 실패: {exc}"
    return templates.TemplateResponse(
        request, "rag.html",
        {
            "question": q,
            "result": result,
            "error": error,
            "model": s.groq_model,
            "key_configured": bool(s.groq_api_key),
            "top_k": s.rag_top_k,
        },
    )


@router.get("/jobs")
async def jobs(request: Request, db: DbDep):
    jobs_rows = (await db.scalars(
        select(IngestJob).order_by(IngestJob.started_at.desc()).limit(50)
    )).all()
    failures = (await db.scalars(
        select(IngestJobItem).where(IngestJobItem.status == "failed")
        .order_by(IngestJobItem.updated_at.desc()).limit(50)
    )).all()
    return templates.TemplateResponse(
        request, "jobs.html",
        {"jobs": jobs_rows, "recent_failures": failures},
    )
