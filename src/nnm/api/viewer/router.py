from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select

from nnm.api.deps import DbDep
from nnm.db.models import Chunk, IngestJob, Paper

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
