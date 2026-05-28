"""평가 러너 — 골든셋 로드, RAG 실행, 자체 구현 4지표 평가.

ragas/langchain 의존성 없음. 모든 LLM 호출은 GroqClient 로 직접.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from nnm.config import Settings
from nnm.db.session import get_factory
from nnm.infra.local_embedder import LocalEmbedder
from nnm.services.rag import RagAnswer, answer

log = structlog.get_logger()


@dataclass
class GoldenItem:
    question: str
    ground_truth: str
    ground_truth_chunk_ids: list[int] | None = None


@dataclass
class EvalSample:
    """RAGAS Dataset 한 행에 대응."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            items.append(
                GoldenItem(
                    question=obj["question"],
                    ground_truth=obj["ground_truth"],
                    ground_truth_chunk_ids=obj.get("ground_truth_chunk_ids"),
                )
            )
    return items


async def run_rag(
    settings: Settings, embedder: LocalEmbedder, item: GoldenItem,
    *, rag_client: Any = None, rag_model_label: str | None = None,
) -> EvalSample:
    factory = get_factory()
    async with factory() as session:
        res: RagAnswer = await answer(
            session, embedder, settings, item.question,
            client=rag_client, model_label=rag_model_label,
        )
    contexts = [c.text for c in res.chunks]
    return EvalSample(
        question=item.question,
        answer=res.answer,
        contexts=contexts,
        ground_truth=item.ground_truth,
    )


async def collect_samples(
    settings: Settings, embedder: LocalEmbedder, items: Iterable[GoldenItem],
    *, rag_client: Any = None, rag_model_label: str | None = None,
) -> list[EvalSample]:
    items = list(items)
    samples: list[EvalSample] = []
    for i, it in enumerate(items, 1):
        log.info("eval.run_rag", idx=i, total=len(items), q=it.question[:60])
        try:
            samples.append(await run_rag(
                settings, embedder, it,
                rag_client=rag_client, rag_model_label=rag_model_label,
            ))
        except Exception as e:  # noqa: BLE001
            log.warning("eval.rag_failed", idx=i, error=str(e))
            samples.append(
                EvalSample(
                    question=it.question,
                    answer="",
                    contexts=[],
                    ground_truth=it.ground_truth,
                )
            )
    return samples


_ALL_METRICS = (
    "faithfulness",
    "context_recall",
    "context_precision",
    "answer_relevancy",
)


def evaluate_samples(
    samples: list[EvalSample],
    settings: Settings,
    *,
    metric_names: list[str] | None = None,
    serial: bool = True,  # noqa: ARG001 - 호환용, 내부에서 항상 직렬
    throttle_s: float = 1.5,
    embedder: LocalEmbedder | None = None,
) -> dict[str, Any]:
    """경량 자체 구현 4지표로 평가 (ragas/langchain 의존성 없음).

    각 metric 은 sample 당 Groq 1호출 이하 — 자세한 호출량은 metrics.py 헤더 참조.
    embedder 인자: GPU/RAM 중복 로드 방지를 위해 collect 단계의 embedder 재사용 권장.
    """
    import os

    from nnm.eval.metrics import MetricResult, evaluate_sample

    metric_names = list(metric_names) if metric_names else list(_ALL_METRICS)
    unknown = [m for m in metric_names if m not in _ALL_METRICS]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}. valid={_ALL_METRICS}")

    # Judge provider 선택 — AWS_BEDROCK_API_KEY 있으면 Bedrock 우선.
    bedrock_key = os.environ.get("AWS_BEDROCK_API_KEY")
    if bedrock_key:
        from nnm.infra.bedrock_client import BedrockClient

        model = os.environ.get(
            "NNM_BEDROCK_MODEL",
            # apac. 접두사 = ap-northeast-2 cross-region inference profile.
            "apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        region = os.environ.get("NNM_BEDROCK_REGION", "ap-northeast-2")
        client: Any = BedrockClient(api_key=bedrock_key, model=model, region=region)
        log.info("eval.judge_provider", provider="bedrock", model=model, region=region)
    else:
        from nnm.infra.groq_client import GroqClient

        if not settings.groq_api_key:
            raise RuntimeError("NNM_GROQ_API_KEY 또는 AWS_BEDROCK_API_KEY 필요")
        client = GroqClient(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            base_url=settings.groq_base_url,
        )
        log.info("eval.judge_provider", provider="groq", model=settings.groq_model)
    if embedder is None:
        embedder = LocalEmbedder(settings=settings)

    async def _run() -> list[MetricResult]:
        results: list[MetricResult] = []
        for i, s in enumerate(samples, 1):
            log.info("eval.sample", idx=i, total=len(samples), q=s.question[:60])
            r = await evaluate_sample(
                client, embedder,
                question=s.question, answer=s.answer,
                contexts=s.contexts, ground_truth=s.ground_truth,
                metrics=metric_names, throttle_s=throttle_s,
            )
            results.append(r)
        return results

    metric_results = asyncio.run(_run())

    # ragas 와 동일한 형태의 DataFrame 출력 (viz 가 그대로 동작하도록).
    import pandas as pd

    rows = []
    for s, m in zip(samples, metric_results, strict=True):
        rows.append({
            "question": s.question,
            "answer": s.answer,
            "contexts": s.contexts,
            "ground_truth": s.ground_truth,
            "faithfulness": m.faithfulness if m.faithfulness is not None else float("nan"),
            "context_recall": m.context_recall if m.context_recall is not None else float("nan"),
            "context_precision": m.context_precision if m.context_precision is not None else float("nan"),
            "answer_relevancy": m.answer_relevancy if m.answer_relevancy is not None else float("nan"),
        })
    df = pd.DataFrame(rows)

    class _Bundle:
        def to_pandas(self) -> Any:
            return df

    return {"result": _Bundle(), "dataset": df, "metric_names": metric_names}


def sync_collect_samples(
    settings: Settings, embedder: LocalEmbedder, items: list[GoldenItem],
    *, rag_client: Any = None, rag_model_label: str | None = None,
) -> list[EvalSample]:
    return asyncio.run(collect_samples(
        settings, embedder, items,
        rag_client=rag_client, rag_model_label=rag_model_label,
    ))
