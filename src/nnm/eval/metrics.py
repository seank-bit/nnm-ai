"""경량 RAGAS-호환 평가 지표 — GroqClient + LocalEmbedder 만으로 동작.

호출량 (샘플당):
- faithfulness:       Groq 1회
- context_recall:     Groq 1회
- context_precision:  Groq 1회 (RAGAS 표준 k회 대비 절감)
- answer_relevancy:   Groq 0회 (bge-m3 코사인 유사도만)

표준 RAGAS 대비 절대 점수는 미세하게 다를 수 있으나 회귀 추이 비교용으로 충분.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from nnm.infra.local_embedder import LocalEmbedder

log = structlog.get_logger()

_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)s")


async def _llm_call(
    client: Any, messages: list[dict[str, str]],
    *, temperature: float = 0.0, max_retries: int = 6,
) -> str:
    """공급자 무관 LLM 호출 wrapper. 429 retry-after 파싱 + 지수 백오프.

    GroqClient / BedrockClient 모두 동일한 chat(messages, temperature) 인터페이스라 swap 가능.
    """
    for attempt in range(max_retries):
        try:
            return await client.chat(messages, temperature=temperature)
        except Exception as e:  # noqa: BLE001 — 공급자 무관 처리
            msg = str(e)
            if "429" not in msg and "throttl" not in msg.lower():
                raise
            m = _RETRY_AFTER_RE.search(msg)
            wait = float(m.group(1)) + 0.5 if m else 2.0 * (attempt + 1)
            log.info(
                "metric.rate_limited", attempt=attempt + 1, wait_s=round(wait, 2),
            )
            await asyncio.sleep(wait)
    raise RuntimeError("max retries exceeded")


def _parse_score(text: str) -> float | None:
    """LLM 응답에서 0~1 점수 추출."""
    m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    m = re.search(r'\b(0?\.\d+|[01])\b', text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return None


FAITHFULNESS_PROMPT = """다음 RAG 답변이 컨텍스트에 충실한지 평가하세요.

[컨텍스트]
{context}

[답변]
{answer}

답변의 모든 사실 주장이 컨텍스트로 직접 뒷받침되는 비율을 0~1 사이로 평가:
- 1.0 = 모든 주장이 컨텍스트로 직접 지지
- 0.5 = 절반 정도만 지지
- 0.0 = 환각/무관

JSON 한 객체만 (백틱 금지): {{"score": 0.0~1.0, "reason": "한 줄 근거"}}"""


async def faithfulness(
    client: Any, *, context: str, answer: str,
) -> tuple[float, str]:
    prompt = FAITHFULNESS_PROMPT.format(
        context=context[:6000], answer=answer[:3000],
    )
    reply = await _llm_call(
        client, [{"role": "user", "content": prompt}],
    )
    return _parse_score(reply) or 0.0, reply[:300]


CONTEXT_RECALL_PROMPT = """ground_truth 가 컨텍스트로부터 얼마나 추론 가능한지 평가.

[컨텍스트]
{context}

[ground_truth]
{ground_truth}

ground_truth 의 각 사실/주장이 컨텍스트에서 추론 가능한 비율을 0~1:
- 1.0 = 모든 내용이 컨텍스트에 있음
- 0.0 = 컨텍스트만으론 추론 불가

JSON: {{"score": 0.0~1.0, "reason": "근거"}}"""


async def context_recall(
    client: Any, *, context: str, ground_truth: str,
) -> tuple[float, str]:
    prompt = CONTEXT_RECALL_PROMPT.format(
        context=context[:6000], ground_truth=ground_truth[:2000],
    )
    reply = await _llm_call(
        client, [{"role": "user", "content": prompt}],
    )
    return _parse_score(reply) or 0.0, reply[:300]


CONTEXT_PRECISION_PROMPT = """ground_truth 와 K개 컨텍스트(검색 순위 순)가 있습니다. 각 컨텍스트가 ground_truth 답변에 관련있는지 0 또는 1로 판정하세요.

[ground_truth]
{ground_truth}

[컨텍스트들]
{contexts_numbered}

K={k} 개의 컨텍스트 각각에 대해 1(관련) 또는 0(무관). 순서대로 배열로 출력.
JSON 한 객체: {{"relevant": [0/1, 0/1, ...]}}"""


async def context_precision(
    client: Any, *, contexts: list[str], ground_truth: str,
) -> tuple[float, str]:
    if not contexts:
        return 0.0, "no contexts"
    k = len(contexts)
    numbered = "\n\n".join(f"[#{i + 1}]\n{c[:1500]}" for i, c in enumerate(contexts))
    prompt = CONTEXT_PRECISION_PROMPT.format(
        ground_truth=ground_truth[:2000],
        contexts_numbered=numbered,
        k=k,
    )
    reply = await _llm_call(
        client, [{"role": "user", "content": prompt}],
    )
    m = re.search(r'"relevant"\s*:\s*\[([^\]]+)\]', reply)
    if not m:
        return 0.0, reply[:300]
    try:
        vals = [
            int(v.strip()) for v in m.group(1).split(",")
            if v.strip() in ("0", "1")
        ]
    except ValueError:
        return 0.0, reply[:300]
    if not vals:
        return 0.0, reply[:300]
    total_rel = sum(vals)
    if total_rel == 0:
        return 0.0, f"all irrelevant; raw={reply[:200]}"
    # average precision @ k = sum(precision@i * rel_i) / total_relevant
    score = 0.0
    cum = 0
    for i, rel in enumerate(vals, 1):
        if rel:
            cum += 1
            score += cum / i
    score /= total_rel
    return score, reply[:300]


async def answer_relevancy(
    embedder: LocalEmbedder, *, question: str, answer: str,
) -> tuple[float, str]:
    """질문/답변 임베딩 코사인 — LLM 호출 0."""
    if not answer.strip():
        return 0.0, "empty answer"
    payload = await embedder.embed(
        [question, answer],
        return_dense=True, return_sparse=False, return_colbert=False,
    )
    q = np.asarray(payload.dense[0], dtype=float)
    a = np.asarray(payload.dense[1], dtype=float)
    denom = float(np.linalg.norm(q) * np.linalg.norm(a)) + 1e-9
    cos = float(np.dot(q, a) / denom)
    score = max(0.0, min(1.0, (cos + 1.0) / 2.0))
    return score, f"cos={cos:.4f}"


@dataclass
class MetricResult:
    faithfulness: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    answer_relevancy: float | None = None
    raw: dict[str, str] = field(default_factory=dict)


async def evaluate_sample(
    client: Any,
    embedder: LocalEmbedder,
    *,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    metrics: list[str],
    throttle_s: float = 1.5,
) -> MetricResult:
    """샘플 1건 평가. 선택한 metric 만 호출."""
    res = MetricResult()
    ctx_joined = "\n\n---\n\n".join(contexts)

    if "faithfulness" in metrics:
        try:
            s, raw = await faithfulness(client, context=ctx_joined, answer=answer)
            res.faithfulness = s
            res.raw["faithfulness"] = raw
        except Exception as e:  # noqa: BLE001
            log.warning("metric.failed", metric="faithfulness", error=str(e))
        await asyncio.sleep(throttle_s)

    if "context_recall" in metrics:
        try:
            s, raw = await context_recall(
                client, context=ctx_joined, ground_truth=ground_truth,
            )
            res.context_recall = s
            res.raw["context_recall"] = raw
        except Exception as e:  # noqa: BLE001
            log.warning("metric.failed", metric="context_recall", error=str(e))
        await asyncio.sleep(throttle_s)

    if "context_precision" in metrics:
        try:
            s, raw = await context_precision(
                client, contexts=contexts, ground_truth=ground_truth,
            )
            res.context_precision = s
            res.raw["context_precision"] = raw
        except Exception as e:  # noqa: BLE001
            log.warning("metric.failed", metric="context_precision", error=str(e))
        await asyncio.sleep(throttle_s)

    if "answer_relevancy" in metrics:
        try:
            s, raw = await answer_relevancy(
                embedder, question=question, answer=answer,
            )
            res.answer_relevancy = s
            res.raw["answer_relevancy"] = raw
        except Exception as e:  # noqa: BLE001
            log.warning("metric.failed", metric="answer_relevancy", error=str(e))

    return res
