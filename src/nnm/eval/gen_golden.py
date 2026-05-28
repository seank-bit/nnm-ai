"""Synthetic 골든셋 생성기.

코퍼스에서 chunk 를 샘플링한 뒤 LLM 으로 (question, ground_truth) 쌍을 생성합니다.
RAGAS 평가에 필요한 골든셋을 자동으로 만드는 기초안 — 사람이 검수/편집 권장.

사용 예:
    python -m nnm.eval.gen_golden \\
        --n 30 --min-chars 400 --out tests/eval/golden.jsonl

요구사항:
    pip install -e ".[eval]"
    NNM_GROQ_API_KEY 설정
    DB 가 실행 중이고 chunks 가 적재돼 있어야 함
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import structlog
import typer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nnm.config import Settings, get_settings
from nnm.db.models import Chunk, Paper
from nnm.db.session import get_factory
from nnm.infra.groq_client import GroqClient, GroqError
from nnm.logging import configure_logging

_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)s")

app = typer.Typer(help="Synthetic 골든셋 생성기", add_completion=False)
log = structlog.get_logger()


GEN_SYSTEM_PROMPT = (
    "당신은 학술 논문 본문 한 조각을 보고, 그 조각만 읽어도 답할 수 있는 "
    "한국어 평가 질문과 정답을 만드는 평가자입니다. "
    "출력은 JSON 한 객체만 반환하세요. 코드블록 백틱 금지."
)

GEN_USER_TEMPLATE = """다음은 논문 \"{title}\" 의 한 부분입니다 (section: {section}).

---
{text}
---

요구사항:
1. 위 본문만으로 답할 수 있는 자연어 질문 1개를 만드세요.
2. 본문에 명시적으로 등장하는 정보로만 정답을 작성하세요 (1~3문장).
3. \"이 논문\", \"본 연구\" 같은 표현은 피하고, 주제어를 그대로 사용하세요.
4. 너무 일반적인 질문 (\"무엇에 관한 글인가?\")은 피하세요.
5. 출력 형식 (JSON 한 객체):
{{
  \"question\": \"...\",
  \"ground_truth\": \"...\"
}}"""


@dataclass
class ChunkRow:
    chunk_id: int
    paper_id: int
    title: str | None
    section: str | None
    text: str


async def sample_chunks(
    db: AsyncSession, *, n: int, min_chars: int, max_chars: int, seed: int
) -> list[ChunkRow]:
    stmt = (
        select(Chunk.id, Chunk.paper_id, Chunk.section, Chunk.text, Paper.title)
        .join(Paper, Paper.id == Chunk.paper_id)
        .where(Chunk.char_count >= min_chars)
        .where(Chunk.char_count <= max_chars)
        .order_by(func.random())
        .limit(n * 3)
    )
    rows = (await db.execute(stmt)).all()
    random.seed(seed)
    random.shuffle(rows)
    seen_papers: set[int] = set()
    selected: list[ChunkRow] = []
    for r in rows:
        if r.paper_id in seen_papers:
            continue
        seen_papers.add(r.paper_id)
        selected.append(
            ChunkRow(
                chunk_id=r.id,
                paper_id=r.paper_id,
                title=r.title,
                section=r.section,
                text=r.text,
            )
        )
        if len(selected) >= n:
            break
    if len(selected) < n:
        for r in rows:
            if any(s.chunk_id == r.id for s in selected):
                continue
            selected.append(
                ChunkRow(
                    chunk_id=r.id, paper_id=r.paper_id, title=r.title,
                    section=r.section, text=r.text,
                )
            )
            if len(selected) >= n:
                break
    return selected[:n]


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(s: str) -> dict[str, str] | None:
    m = _JSON_BLOCK_RE.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    q = (obj.get("question") or "").strip()
    a = (obj.get("ground_truth") or obj.get("answer") or "").strip()
    if not q or not a:
        return None
    return {"question": q, "ground_truth": a}


async def gen_for_chunk(
    client: GroqClient, settings: Settings, row: ChunkRow, max_text_chars: int,
    max_retries: int = 5,
) -> dict[str, str] | None:
    text = row.text.strip()
    if len(text) > max_text_chars:
        text = text[:max_text_chars] + " …"
    prompt = GEN_USER_TEMPLATE.format(
        title=(row.title or "(제목없음)")[:120],
        section=row.section or "-",
        text=text,
    )
    messages = [
        {"role": "system", "content": GEN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(max_retries):
        try:
            reply = await client.chat(messages, temperature=0.4)
            return _extract_json(reply)
        except GroqError as e:
            msg = str(e)
            if "429" not in msg:
                log.warning("gen.llm_failed", chunk_id=row.chunk_id, error=msg)
                return None
            # 429 — Groq 가 "try again in Xs" 를 본문에 포함. 못 찾으면 지수 백오프.
            m = _RETRY_AFTER_RE.search(msg)
            wait = float(m.group(1)) + 0.5 if m else 2.0 * (attempt + 1)
            log.info(
                "gen.rate_limited", chunk_id=row.chunk_id,
                attempt=attempt + 1, wait_s=round(wait, 2),
            )
            await asyncio.sleep(wait)
        except Exception as e:  # noqa: BLE001
            log.warning("gen.llm_failed", chunk_id=row.chunk_id, error=str(e))
            return None
    log.warning("gen.retry_exhausted", chunk_id=row.chunk_id)
    return None


async def run(
    n: int, min_chars: int, max_chars: int, max_text_chars: int,
    seed: int, out_path: Path,
) -> None:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("NNM_GROQ_API_KEY 미설정")
    factory = get_factory()
    async with factory() as db:
        rows = await sample_chunks(
            db, n=n, min_chars=min_chars, max_chars=max_chars, seed=seed
        )
    if not rows:
        raise RuntimeError("샘플링된 chunk 가 0개입니다. DB에 적재된 데이터를 확인하세요.")
    log.info("gen.sampled", n=len(rows))

    client = GroqClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "# Synthetic 골든셋 — 사람이 검수/편집 권장.\n"
            f"# 생성 옵션: n={n} min_chars={min_chars} seed={seed}\n"
        )
        for i, row in enumerate(rows, 1):
            qa = await gen_for_chunk(client, settings, row, max_text_chars)
            if qa is None:
                log.warning("gen.skipped", idx=i, chunk_id=row.chunk_id)
                continue
            obj = {
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "ground_truth_chunk_ids": [row.chunk_id],
                "source_paper_id": row.paper_id,
                "source_title": row.title,
                "source_section": row.section,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            log.info("gen.wrote", idx=i, chunk_id=row.chunk_id, q=qa["question"][:60])
    log.info("gen.done", written=written, path=str(out_path))

    # 검수용 HTML 뷰어 자동 생성 (/eval/golden.html 로 접근).
    try:
        from nnm.eval.viz import render_golden_html, render_index

        html_path = out_path.with_suffix(".html")
        render_golden_html(out_path, html_path)
        render_index(out_path.parent)
        log.info("gen.rendered_html", path=str(html_path))
    except Exception as e:  # noqa: BLE001
        log.warning("gen.render_html_failed", error=str(e))


@app.command()
def main(
    n: int = typer.Option(30, "--n", help="생성할 골든 항목 수"),
    min_chars: int = typer.Option(400, "--min-chars", help="샘플 chunk 최소 문자수"),
    max_chars: int = typer.Option(3000, "--max-chars", help="샘플 chunk 최대 문자수"),
    max_text_chars: int = typer.Option(
        2500, "--max-text-chars", help="LLM 에 넣을 본문 길이 상한"
    ),
    seed: int = typer.Option(7, "--seed", help="샘플링 난수 시드"),
    out: Path = typer.Option(
        Path("var/eval/golden.jsonl"), "--out", "-o", help="출력 경로"
    ),
) -> None:
    configure_logging()
    asyncio.run(run(n, min_chars, max_chars, max_text_chars, seed, out))


if __name__ == "__main__":
    app()
