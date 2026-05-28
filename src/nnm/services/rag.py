from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nnm.config import Settings
from nnm.db.models import Chunk, ChunkEmbedding, Paper
from nnm.infra.groq_client import GroqClient
from nnm.infra.local_embedder import LocalEmbedder

log = structlog.get_logger()

SYSTEM_PROMPT = (
    "당신은 학술 논문 검색 결과를 바탕으로 답변하는 보조자입니다. "
    "주어진 컨텍스트 안의 내용만 근거로 사용하고, 근거가 부족하면 모른다고 답하세요. "
    "각 주장 끝에 [#chunk_id] 형태로 출처를 표시하세요."
)


@dataclass
class RetrievedChunk:
    chunk_id: int
    paper_id: int
    paper_title: str | None
    section: str | None
    seq: int
    text: str
    score: float


@dataclass
class RagAnswer:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    model: str


async def retrieve(
    db: AsyncSession,
    embedder: LocalEmbedder,
    question: str,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    payload = await embedder.embed(
        [question], return_dense=True, return_sparse=False, return_colbert=False,
    )
    qvec = payload.dense[0].tolist()
    distance = ChunkEmbedding.dense.cosine_distance(qvec).label("distance")
    stmt = (
        select(
            ChunkEmbedding.chunk_id,
            distance,
            Chunk.paper_id,
            Chunk.seq,
            Chunk.section,
            Chunk.text,
            Paper.title,
        )
        .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
        .join(Paper, Paper.id == Chunk.paper_id)
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await db.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=r.chunk_id,
            paper_id=r.paper_id,
            paper_title=r.title,
            section=r.section,
            seq=r.seq,
            text=r.text,
            score=1.0 - float(r.distance),
        )
        for r in rows
    ]


def _build_context(chunks: list[RetrievedChunk], max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for c in chunks:
        header = (
            f"[#{c.chunk_id}] paper={c.paper_id} "
            f"title={(c.paper_title or '')[:80]} section={c.section or '-'}"
        )
        body = c.text.strip()
        remaining = max_chars - total
        if remaining <= len(header) + 8:
            break
        if len(body) > remaining - len(header) - 8:
            body = body[: remaining - len(header) - 8] + " …"
        block = f"{header}\n{body}"
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


async def answer(
    db: AsyncSession,
    embedder: LocalEmbedder,
    settings: Settings,
    question: str,
    *,
    client: Any = None,
    model_label: str | None = None,
) -> RagAnswer:
    """RAG 답변 생성.

    client 미지정 시 GroqClient 기본 사용. 평가용으로 BedrockClient 같은 다른 공급자 주입 가능.
    """
    chunks = await retrieve(db, embedder, question, top_k=settings.rag_top_k)
    context = _build_context(chunks, settings.rag_max_context_chars)
    user_prompt = (
        f"질문:\n{question}\n\n"
        f"컨텍스트:\n{context if context else '(검색된 내용 없음)'}\n\n"
        "위 컨텍스트만 사용해 한국어로 답하세요."
    )
    if client is None:
        client = GroqClient(
            api_key=settings.groq_api_key or "",
            model=settings.groq_model,
            base_url=settings.groq_base_url,
        )
        model_label = settings.groq_model
    reply = await client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.rag_temperature,
    )
    return RagAnswer(
        question=question, answer=reply, chunks=chunks,
        model=model_label or getattr(client, "model", "unknown"),
    )
