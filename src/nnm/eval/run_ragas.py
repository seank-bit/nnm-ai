"""RAGAS 평가 CLI.

사용 예:
    python -m nnm.eval.run_ragas \\
        --golden tests/eval/golden.jsonl \\
        --out var/eval

요구사항:
    pip install -e .[eval]
    NNM_GROQ_API_KEY 또는 OPENAI_API_KEY 설정
    DB 가 실행 중이고 ChunkEmbedding 이 채워져 있어야 함
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog
import typer

from nnm.config import get_settings
from nnm.eval.ragas_runner import (
    evaluate_samples,
    load_golden,
    sync_collect_samples,
)
from nnm.eval.viz import render_charts_png, render_html_report, render_index
from nnm.infra.local_embedder import LocalEmbedder
from nnm.logging import configure_logging

app = typer.Typer(help="RAGAS 평가 러너", add_completion=False)
log = structlog.get_logger()

METRICS = ("faithfulness", "context_recall", "context_precision", "answer_relevancy")
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.70,
    "context_recall": 0.60,
    "context_precision": 0.60,
    "answer_relevancy": 0.70,
}


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# 지표당 sample 당 평균 judge 호출 수 (대략치). preflight 추정용.
_CALLS_PER_SAMPLE = {
    "faithfulness": 4,       # claim 추출 + 각 claim 검증
    "context_recall": 3,     # ground_truth 문장 분해 + 각 문장 검증
    "context_precision": 5,  # top_k(=5) 각 context 관련성 평가
    "answer_relevancy": 3,   # reverse question 3개 생성
}


def _estimate_calls(n_samples: int, metrics: list[str]) -> int:
    # RAG 답변 생성 1 호출 + 각 지표별 judge 호출.
    judge = sum(_CALLS_PER_SAMPLE.get(m, 0) for m in metrics)
    return n_samples * (1 + judge)


@app.command()
def main(
    golden: Path = typer.Option(
        Path("var/eval/golden.jsonl"), "--golden", "-g", help="골든셋 JSONL 경로"
    ),
    out_dir: Path = typer.Option(
        Path("var/eval"), "--out", "-o", help="결과 CSV/JSON 출력 디렉토리"
    ),
    limit: int | None = typer.Option(None, "--limit", help="앞에서 N개만 평가"),
    metrics_csv: str = typer.Option(
        ",".join(METRICS), "--metrics", "-m",
        help="평가할 지표 (comma-separated). 호출량 절감 목적으로 일부 제외 가능",
    ),
    enforce: bool = typer.Option(
        False, "--enforce", help="임계값 미달 시 exit code 1"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="사전 호출량 안내 후 확인 프롬프트 생략",
    ),
) -> None:
    configure_logging()
    settings = get_settings()

    if not golden.exists():
        typer.secho(f"골든셋 파일 없음: {golden}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    items = load_golden(golden)
    if limit is not None:
        items = items[:limit]
    if not items:
        typer.secho("골든셋이 비어 있음", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    selected_metrics = [m.strip() for m in metrics_csv.split(",") if m.strip()]
    unknown = [m for m in selected_metrics if m not in METRICS]
    if unknown:
        typer.secho(f"알 수 없는 지표: {unknown}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    est = _estimate_calls(len(items), selected_metrics)
    typer.secho("\n=== 사전 호출량 추정 ===", bold=True)
    typer.echo(f"  샘플 수      : {len(items)}")
    typer.echo(f"  지표         : {', '.join(selected_metrics)}")
    typer.echo(f"  예상 Groq 호출: 약 {est}회 (RAG 답변 {len(items)}회 + judge ~{est - len(items)}회)")
    typer.echo("  TPM 보호    : 직렬 실행 + 429 자동 재시도")
    if not yes:
        if not typer.confirm("진행할까요?", default=False):
            typer.echo("취소됨.")
            raise typer.Exit(0)

    log.info("eval.start", n=len(items), golden=str(golden), metrics=selected_metrics)

    embedder = LocalEmbedder(settings=settings)

    # Groq TPD 소진 등으로 Groq 답변 생성이 막힐 때 Bedrock 으로 fallback.
    # AWS_BEDROCK_API_KEY 환경변수가 있고 NNM_EVAL_RAG_PROVIDER=bedrock 이면 사용.
    bedrock_key = os.environ.get("AWS_BEDROCK_API_KEY")
    rag_client = None
    rag_model_label = None
    if bedrock_key and os.environ.get("NNM_EVAL_RAG_PROVIDER", "").lower() == "bedrock":
        from nnm.infra.bedrock_client import BedrockClient
        rag_model = os.environ.get(
            "NNM_BEDROCK_MODEL",
            "apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        rag_region = os.environ.get("NNM_BEDROCK_REGION", "ap-northeast-2")
        rag_client = BedrockClient(api_key=bedrock_key, model=rag_model, region=rag_region)
        rag_model_label = f"bedrock:{rag_model}"
        log.info("eval.rag_provider", provider="bedrock", model=rag_model)
    else:
        log.info("eval.rag_provider", provider="groq", model=settings.groq_model)

    samples = sync_collect_samples(
        settings, embedder, items,
        rag_client=rag_client, rag_model_label=rag_model_label,
    )

    log.info("eval.run_ragas", n=len(samples), metrics=selected_metrics)
    bundle = evaluate_samples(
        samples, settings, metric_names=selected_metrics, embedder=embedder,
    )
    result = bundle["result"]

    # ragas 0.2+ 는 result.to_pandas() 지원.
    try:
        df = result.to_pandas()
    except AttributeError:
        df = bundle["dataset"].to_pandas()
        for m in METRICS:
            if hasattr(result, m):
                df[m] = getattr(result, m)

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"ragas_{ts}.csv"
    json_path = out_dir / f"ragas_{ts}.json"
    png_path = out_dir / f"ragas_{ts}.png"
    html_path = out_dir / f"ragas_{ts}.html"

    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary: dict[str, float] = {}
    for m in METRICS:
        if m in df.columns:
            try:
                summary[m] = float(df[m].mean())
            except Exception:  # noqa: BLE001
                summary[m] = float("nan")

    meta = {
        "timestamp": ts,
        "git_sha": _git_sha(),
        "model": settings.groq_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.rag_top_k,
        "temperature": settings.rag_temperature,
        "n_samples": len(samples),
        "judge_model": os.environ.get("RAGAS_JUDGE_MODEL", settings.groq_model),
        "summary": summary,
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    try:
        render_charts_png(df, png_path)
        render_html_report(df, meta, png_path, html_path)
        render_index(out_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("eval.viz_failed", error=str(e))
        png_path = html_path = None  # type: ignore[assignment]

    typer.secho("\n=== RAGAS 결과 요약 ===", bold=True)
    for m in METRICS:
        v = summary.get(m, float("nan"))
        typer.echo(f"  {m:>20s}: {v:.4f}")
    typer.echo(f"\n저장: {csv_path}")
    typer.echo(f"      {json_path}")
    if png_path:
        typer.echo(f"      {png_path}")
    if html_path:
        typer.echo(f"      {html_path}  (브라우저로 열어 차트+표 확인)")

    if enforce:
        bad = {
            m: (summary.get(m), DEFAULT_THRESHOLDS[m])
            for m in METRICS
            if summary.get(m, 0.0) < DEFAULT_THRESHOLDS[m]
        }
        if bad:
            typer.secho("\n임계값 미달:", fg=typer.colors.RED, err=True)
            for m, (got, th) in bad.items():
                typer.secho(f"  {m}: {got:.4f} < {th}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
    sys.exit(0)
