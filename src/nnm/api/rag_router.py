"""JSON RAG API — 다른 서비스에서 호출하기 위한 엔드포인트.

POST /api/rag
Content-Type: application/json
{"question": "..."}

→ 200 OK
{"question": "...", "answer": "...", "model": "...", "chunks": [...]}
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from nnm.api.deps import DbDep, EmbedderDep
from nnm.config import get_settings
from nnm.infra.groq_client import GroqError
from nnm.services import rag as rag_service

router = APIRouter()
log = structlog.get_logger()


class RagRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=4000,
        description="사용자 자연어 질문 (한국어 권장)",
    )


class RagChunk(BaseModel):
    chunk_id: int
    paper_id: int
    paper_title: str | None
    section: str | None
    seq: int
    score: float = Field(..., description="유사도 점수 (0~1, 클수록 관련)")
    text: str
    s3_key: str | None = Field(
        None,
        description="PDF s3_key 에서 설정된 PDF prefix (예: 'newnonmuncom-pdf/') 제거한 값. "
                    "클라이언트가 별도 prefix 와 합쳐 다운로드 URL 등 구성 가능.",
    )


class RagResponse(BaseModel):
    question: str
    answer: str
    model: str = Field(..., description="답변 생성 LLM 모델 식별자")
    chunks: list[RagChunk] = Field(..., description="검색된 컨텍스트 chunk (rank 순)")


@router.post(
    "/rag",
    response_model=RagResponse,
    summary="RAG 질문 → 답변 + 출처 chunk 반환",
    responses={
        400: {"description": "잘못된 요청 (빈 질문 등)"},
        503: {"description": "LLM 호출 실패 (Groq API 에러 / API key 미설정)"},
    },
)
async def rag_query(
    body: RagRequest, db: DbDep, embedder: EmbedderDep,
) -> RagResponse:
    s = get_settings()
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question 이 비어 있습니다.")
    if not s.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="NNM_GROQ_API_KEY 미설정 — 서버 환경변수를 확인하세요.",
        )
    try:
        result = await rag_service.answer(db, embedder, s, q)
    except GroqError as exc:
        log.warning("api.rag.groq_error", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Groq API 에러: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("api.rag.failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"RAG 실행 실패: {exc}") from exc

    pdf_prefix = s.s3_pdf_prefix or s.s3_prefix or ""
    return RagResponse(
        question=result.question,
        answer=result.answer,
        model=result.model,
        chunks=[
            RagChunk(
                chunk_id=c.chunk_id,
                paper_id=c.paper_id,
                paper_title=c.paper_title,
                section=c.section,
                seq=c.seq,
                score=c.score,
                text=c.text,
                s3_key=_strip_pdf_prefix(c.s3_key, pdf_prefix),
            )
            for c in result.chunks
        ],
    )


def _strip_pdf_prefix(s3_key: str | None, prefix: str) -> str | None:
    if s3_key is None:
        return None
    if prefix and s3_key.startswith(prefix):
        return s3_key[len(prefix):]
    return s3_key
