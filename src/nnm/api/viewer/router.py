from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select

from nnm.api.deps import DbDep
from nnm.db.models import Chunk, ChunkEmbedding, IngestJob, IngestJobItem, Paper
from nnm.errors import PaperNotFound

router = APIRouter()
_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))

PAGE_SIZE = 50


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

    recent_jobs = (await db.scalars(
        select(IngestJob).order_by(IngestJob.started_at.desc()).limit(5)
    )).all()

    return templates.TemplateResponse(
        request, "dashboard.html",
        {
            "total_papers": total_papers, "total_chunks": total_chunks,
            "by_status": by_status, "mapping_rate": mapping_rate,
            "recent_jobs": recent_jobs,
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
