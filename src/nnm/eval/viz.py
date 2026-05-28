"""RAGAS 결과 시각화 — 막대/레이더/히트맵 + HTML 리포트.

[eval] extras 의 matplotlib 필요.
"""

from __future__ import annotations

import base64
import html
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("faithfulness", "context_recall", "context_precision", "answer_relevancy")
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
    "answer_relevancy": "Answer Relevance",
}
METRIC_DESCRIPTIONS = {
    "faithfulness": "답변이 컨텍스트에 충실한 정도 (환각 여부)",
    "context_recall": "정답 정보가 컨텍스트에 얼마나 회수됐는지",
    "context_precision": "검색된 컨텍스트 중 관련 있는 것이 상위에 있는지",
    "answer_relevancy": "답변이 질문과 의미적으로 일치하는 정도",
}


def _metric_means(df: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in METRICS:
        if m in df.columns:
            try:
                out[m] = float(df[m].mean())
            except Exception:  # noqa: BLE001
                out[m] = float("nan")
        else:
            out[m] = float("nan")
    return out


# 모던하고 가독성 좋은 색상 팔레트 (각 metric 별 고유 색).
_METRIC_COLORS = {
    "faithfulness": "#4F46E5",       # indigo
    "context_recall": "#F59E0B",     # amber
    "context_precision": "#10B981",  # emerald
    "answer_relevancy": "#EF4444",   # red
}


# --- HTML 리포트 공용 디자인 시스템 (모든 페이지가 동일 CSS 변수 사용) ---

_SHARED_CSS = """
:root {
  --bg: #F9FAFB; --surface: #FFFFFF; --border: #E5E7EB; --border-strong: #D1D5DB;
  --text: #111827; --text-muted: #6B7280; --text-subtle: #9CA3AF;
  --primary: #4F46E5;
  --good: #10B981; --good-bg: #ECFDF5;
  --mid: #F59E0B; --mid-bg: #FFFBEB;
  --bad: #EF4444; --bad-bg: #FEF2F2;
  --shadow-sm: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05);
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Pretendard', 'Noto Sans KR', Roboto, sans-serif;
  margin: 0; padding: 32px 40px; background: var(--bg); color: var(--text);
  font-size: 14px; line-height: 1.5;
}
h1 { font-size: 24px; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.01em; }
h2 { font-size: 16px; font-weight: 600; margin: 32px 0 12px 0; color: var(--text); }
.subtitle { color: var(--text-muted); margin: 0 0 24px 0; font-size: 13px; }
.subtitle code { background: #F3F4F6; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px; box-shadow: var(--shadow-sm);
}
.tile-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;
}
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; position: relative; overflow: hidden; box-shadow: var(--shadow-sm); }
.tile::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: var(--tile-color, var(--primary));
}
.tile-label { color: var(--text-muted); font-size: 12px; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.04em; }
.tile-value { font-size: 32px; font-weight: 700; margin: 4px 0 8px 0;
  font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.tile-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.tile-bar-fill { height: 100%; background: var(--tile-color, var(--primary)); transition: width .3s; }
.tile-desc { color: var(--text-muted); font-size: 12px; margin-top: 10px; line-height: 1.45; }
.insights {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 22px; margin-bottom: 24px; box-shadow: var(--shadow-sm);
}
.insights h3 { font-size: 14px; font-weight: 600; margin: 0 0 10px 0; color: var(--text); }
.insights p { margin: 0 0 8px 0; color: var(--text); font-size: 13px; line-height: 1.6; }
.insights .pill {
  display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11px;
  font-weight: 600; vertical-align: middle; margin: 0 1px;
}
.insights .pill.good { background: var(--good-bg); color: var(--good); }
.insights .pill.mid  { background: var(--mid-bg);  color: #B45309; }
.insights .pill.bad  { background: var(--bad-bg);  color: var(--bad); }
.insights ul { margin: 6px 0 0 0; padding-left: 20px; color: var(--text-muted); font-size: 13px; }
.insights ul li { margin: 3px 0; line-height: 1.5; }
.insights ul li strong { color: var(--text); font-weight: 600; }
.meta-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px 24px; font-size: 13px;
}
.meta-grid dt { color: var(--text-muted); font-size: 11px; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px; }
.meta-grid dd { margin: 0 0 8px 0; color: var(--text); font-variant-numeric: tabular-nums; }
img.chart { width: 100%; height: auto; border-radius: 8px; display: block; }
.table-wrap {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden; box-shadow: var(--shadow-sm);
}
table.data { width: 100%; border-collapse: collapse; font-size: 13px; }
table.data thead th {
  background: #F3F4F6; color: var(--text-muted); font-weight: 600;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
  padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border-strong);
  position: sticky; top: 0; z-index: 1;
}
table.data tbody td {
  padding: 12px 14px; border-bottom: 1px solid var(--border); vertical-align: top;
}
table.data tbody tr:hover { background: #FAFBFC; }
table.data tbody tr:last-child td { border-bottom: none; }
td.idx { color: var(--text-subtle); font-variant-numeric: tabular-nums; width: 40px; }
td.text { max-width: 360px; }
td.text .clamp {
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden; line-height: 1.45;
}
td.score { font-variant-numeric: tabular-nums; font-weight: 600; text-align: right;
  width: 80px; white-space: nowrap; }
td.score.good { color: var(--good); background: var(--good-bg); }
td.score.mid  { color: #B45309; background: var(--mid-bg); }
td.score.bad  { color: var(--bad);  background: var(--bad-bg); }
td.score.empty { color: var(--text-subtle); }
details.contexts { margin-top: 6px; }
details.contexts summary { cursor: pointer; color: var(--primary); font-size: 12px;
  user-select: none; outline: none; }
details.contexts ol { margin: 8px 0 0 0; padding-left: 20px; font-size: 12px;
  color: var(--text-muted); }
details.contexts ol li { margin: 4px 0; line-height: 1.5; }
.copy-hint { color: var(--text-subtle); font-size: 12px; margin: 0 0 8px 0; }
.report-list { list-style: none; padding: 0; margin: 0; }
.report-list li { padding: 14px 16px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 16px; }
.report-list li:last-child { border-bottom: none; }
.report-list .ts { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px; color: var(--primary); font-weight: 500; text-decoration: none;
  flex: 1; }
.report-list .ts:hover { text-decoration: underline; }
.report-list .siblings { display: flex; gap: 8px; }
.report-list .siblings a {
  font-size: 11px; padding: 3px 8px; border-radius: 4px;
  background: #F3F4F6; color: var(--text-muted); text-decoration: none;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.report-list .siblings a:hover { background: var(--border); color: var(--text); }
.empty-state { padding: 48px 20px; text-align: center; color: var(--text-muted); }
.empty-state code { background: #F3F4F6; padding: 2px 8px; border-radius: 4px;
  font-size: 13px; color: var(--text); }
"""


def _score_class(v: float) -> str:
    if np.isnan(v):
        return "empty"
    if v >= 0.8:
        return "good"
    if v >= 0.6:
        return "mid"
    return "bad"


def _truncate(text: str, n: int = 180) -> tuple[str, bool]:
    t = str(text).strip()
    if len(t) > n:
        return t[:n].rstrip() + "…", True
    return t, False


def _pill(score: float) -> str:
    cls = _score_class(score)
    label = "—" if cls == "empty" else f"{score:.3f}"
    return f'<span class="pill {cls}">{label}</span>'


def _build_insights(df: Any, summary: dict[str, float]) -> str:
    """결과 요약 인사이트 — df 와 summary 에서 동적 계산."""
    metric_cols = [m for m in METRICS if m in df.columns]
    if not metric_cols:
        return ""

    # 평균 of 평균 (단일 quality 지표).
    valid_means = [summary.get(m, float("nan")) for m in metric_cols]
    valid_means = [v for v in valid_means if not np.isnan(v)]
    overall = float(np.mean(valid_means)) if valid_means else float("nan")

    # 최강 / 최약 지표.
    sorted_metrics = sorted(
        ((m, summary.get(m, float("nan"))) for m in metric_cols),
        key=lambda x: x[1] if not np.isnan(x[1]) else 0.0,
    )
    weakest = sorted_metrics[0]
    strongest = sorted_metrics[-1]

    # 약점 샘플 — 어느 지표에서든 < 0.5 인 케이스.
    matrix = df[list(metric_cols)].to_numpy(dtype=float)
    weak_rows = []
    for i in range(matrix.shape[0]):
        row_vals = matrix[i]
        bad_metrics = [
            (metric_cols[j], row_vals[j])
            for j in range(len(metric_cols))
            if not np.isnan(row_vals[j]) and row_vals[j] < 0.5
        ]
        if bad_metrics:
            weak_rows.append((i + 1, bad_metrics))
    n_weak = len(weak_rows)
    n_total = matrix.shape[0]
    # 가장 점수 낮은 샘플 3개 (전체 평균 기준).
    sample_avgs = np.nanmean(matrix, axis=1)
    worst_idx = np.argsort(sample_avgs)[:3]

    # ─── 자연어 해석 문구 ───
    def _qual(v: float) -> str:
        if v >= 0.85:
            return "매우 양호"
        if v >= 0.75:
            return "양호"
        if v >= 0.6:
            return "보통"
        return "개선 필요"

    overall_pill = _pill(overall) if not np.isnan(overall) else ""
    strong_pill = _pill(strongest[1])
    weak_pill = _pill(weakest[1])

    summary_sentence = (
        f"30건 평가 기준, 종합 평균 {overall_pill} ({_qual(overall)}). "
        f"가장 강한 지표는 <strong>{METRIC_LABELS[strongest[0]]}</strong> {strong_pill}, "
        f"가장 약한 지표는 <strong>{METRIC_LABELS[weakest[0]]}</strong> {weak_pill}."
    )

    weak_sample_lines: list[str] = []
    for idx in worst_idx:
        i = int(idx) + 1
        row_score = float(sample_avgs[idx])
        if row_score >= 0.7:
            continue  # 전체 평균이 높으면 약점 사례로 노출 안 함
        bad_parts = []
        for j, m in enumerate(metric_cols):
            v = matrix[idx, j]
            if not np.isnan(v) and v < 0.5:
                bad_parts.append(f"{METRIC_LABELS[m]} {v:.2f}")
        bad_str = ", ".join(bad_parts) if bad_parts else f"평균 {row_score:.2f}"
        q_text = ""
        if "question" in df.columns:
            q_text, _ = _truncate(str(df.iloc[idx]["question"]), 70)
        weak_sample_lines.append(
            f"<li><strong>Q{i}</strong> — {html.escape(q_text)}"
            f" <span style='color:var(--bad)'>({bad_str})</span></li>"
        )

    weak_block = ""
    if weak_sample_lines:
        weak_block = (
            f"<p style='margin-top:10px'>약점 샘플 (전체 평균 0.7 미만):</p>"
            f"<ul>{''.join(weak_sample_lines)}</ul>"
        )

    coverage_line = (
        f"전체 {n_total}건 중 어느 한 지표라도 0.5 미만인 샘플: <strong>{n_weak}건</strong>"
        f" ({n_weak * 100 / n_total:.0f}%)."
    ) if n_total else ""

    return f"""
<div class="insights">
  <h3>📊 결과 요약</h3>
  <p>{summary_sentence}</p>
  <p>{coverage_line}</p>
  {weak_block}
</div>
"""


def _setup_modern_style() -> None:
    """matplotlib 글로벌 스타일 — 깔끔하고 모던하게."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#D1D5DB",
        "axes.labelcolor": "#374151",
        "xtick.color": "#6B7280",
        "ytick.color": "#6B7280",
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def render_charts_png(df: Any, out_path: Path) -> Path:
    """가로 막대 차트 + 깔끔한 히트맵 (2단 구성).

    레이더 제거 — 4축에서는 정보량이 막대보다 낮음.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_modern_style()

    means = _metric_means(df)
    cols_present = [m for m in METRICS if m in df.columns]
    n_samples = len(df)

    # 샘플 수에 비례해 히트맵 높이 조절.
    hm_height = max(4.0, 0.25 * n_samples + 1.5)
    fig = plt.figure(figsize=(12, 3.2 + hm_height), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, hm_height / 3.2])

    # ─── (1) 가로 막대 차트 — 4개 평균
    ax_bar = fig.add_subplot(gs[0, 0])
    labels = [METRIC_LABELS[m] for m in METRICS]
    values = [means[m] for m in METRICS]
    colors = [_METRIC_COLORS[m] for m in METRICS]
    y_pos = np.arange(len(METRICS))
    bars = ax_bar.barh(y_pos, values, color=colors, height=0.55, edgecolor="none")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=12)
    ax_bar.invert_yaxis()  # 위→아래 순서
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_bar.set_xlabel("Score")
    ax_bar.set_title("Average Scores", loc="left", pad=12)
    ax_bar.grid(axis="x", alpha=0.4)
    ax_bar.grid(axis="y", visible=False)
    # 값 라벨을 막대 끝 오른쪽에 표시.
    for bar, v in zip(bars, values, strict=False):
        if np.isnan(v):
            continue
        ax_bar.text(
            v + 0.015, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", ha="left",
            fontsize=11, fontweight="600", color="#111827",
        )

    # ─── (2) 히트맵 — 샘플(세로) × 지표(가로). 30+ 샘플 가독성 ↑.
    ax_hm = fig.add_subplot(gs[1, 0])
    if cols_present:
        matrix = df[list(cols_present)].to_numpy(dtype=float)
        im = ax_hm.imshow(
            matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax_hm.set_xticks(range(len(cols_present)))
        ax_hm.set_xticklabels(
            [METRIC_LABELS[m] for m in cols_present],
            fontsize=11,
        )
        ax_hm.set_yticks(range(n_samples))
        ax_hm.set_yticklabels([f"Q{i + 1}" for i in range(n_samples)], fontsize=9)
        ax_hm.set_title("Per-Sample Scores", loc="left", pad=12)
        ax_hm.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
        ax_hm.grid(visible=False)
        cbar = fig.colorbar(im, ax=ax_hm, shrink=0.6, pad=0.02)
        cbar.set_label("score", fontsize=10)
        cbar.outline.set_visible(False)
        # 셀 안에 점수 적기 — 샘플 수에 따라 폰트 사이즈 조절.
        cell_fs = 9 if n_samples <= 15 else 8 if n_samples <= 30 else 7
        for s in range(n_samples):
            for m_idx in range(matrix.shape[1]):
                v = matrix[s, m_idx]
                if np.isnan(v):
                    continue
                ax_hm.text(
                    m_idx, s, f"{v:.2f}", ha="center", va="center",
                    fontsize=cell_fs,
                    color="#111827" if v > 0.5 else "white",
                    fontweight="500",
                )

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _png_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _format_cell(v: Any) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.4f}" if 0 <= v <= 1 else f"{v:g}"
    if isinstance(v, list):
        return " | ".join(str(x) for x in v)
    return html.escape(str(v)) if v is not None else ""


def render_golden_html(jsonl_path: Path, out_html: Path) -> Path:
    """gen_golden 산출물 검수용 HTML — 통합 디자인 시스템 사용."""
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    body_rows: list[str] = []
    for i, r in enumerate(rows, 1):
        q = html.escape(str(r.get("question", "")))
        gt = html.escape(str(r.get("ground_truth", "")))
        title = html.escape(str(r.get("source_title") or "—"))
        section = html.escape(str(r.get("source_section") or ""))
        chunk_ids = r.get("ground_truth_chunk_ids") or []
        chunk_str = ", ".join(str(c) for c in chunk_ids)
        paper_id = r.get("source_paper_id")
        meta_parts = []
        if section:
            meta_parts.append(section)
        if paper_id:
            meta_parts.append(f"paper #{paper_id}")
        if chunk_str:
            meta_parts.append(f"chunks: {chunk_str}")
        meta_line = " · ".join(meta_parts)
        body_rows.append(
            f'<tr>'
            f'<td class="idx">{i}</td>'
            f'<td class="text"><div class="clamp">{q}</div></td>'
            f'<td class="text"><div class="clamp">{gt}</div></td>'
            f'<td class="text">'
            f'  <div style="color:var(--primary);font-weight:500">{title}</div>'
            f'  <div style="color:var(--text-muted);font-size:12px;margin-top:4px">{meta_line}</div>'
            f'</td>'
            f'</tr>'
        )

    n = len(rows)
    empty_row = (
        '<tr><td colspan="4" style="text-align:center;color:var(--text-subtle);padding:48px">'
        '데이터 없음</td></tr>'
    )
    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Golden Set ({n})</title>
<style>{_SHARED_CSS}</style>
</head>
<body>
  <h1>골든셋 ({n}건)</h1>
  <p class="subtitle">
    Groq 가 코퍼스 chunk 에서 자동 생성한 (질문, 정답) 쌍. <strong>검수 후</strong> 평가에 사용.
    원본 파일: <code>{html.escape(str(jsonl_path.name))}</code> ·
    <a href="./" style="color:var(--primary);text-decoration:none">← 리포트 목록</a>
  </p>
  <div class="table-wrap">
    <table class="data">
      <thead><tr>
        <th>#</th><th>Question</th><th>Ground Truth</th><th>Source</th>
      </tr></thead>
      <tbody>{''.join(body_rows) if body_rows else empty_row}</tbody>
    </table>
  </div>
</body></html>
"""
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def render_index(eval_dir: Path) -> Path:
    """eval 디렉토리의 ragas_*.html 목록 — 통합 디자인 시스템 사용."""
    reports = sorted(
        eval_dir.glob("ragas_*.html"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    items: list[str] = []
    for r in reports:
        ts = r.stem.removeprefix("ragas_")
        siblings = "".join(
            f'<a href="{html.escape(s.name)}">{s.suffix.lstrip(".").upper()}</a>'
            for s in (
                eval_dir / f"{r.stem}.csv",
                eval_dir / f"{r.stem}.json",
                eval_dir / f"{r.stem}.png",
            )
            if s.exists()
        )
        items.append(
            f'<li>'
            f'<a class="ts" href="{html.escape(r.name)}">{html.escape(ts)}</a>'
            f'<span class="siblings">{siblings}</span>'
            f'</li>'
        )

    results_body = (
        f'<div class="empty-state">리포트가 아직 없습니다.'
        f' <code>python -m nnm.eval.run_ragas</code> 실행 후 새로고침.</div>'
        if not items
        else f'<ul class="report-list">{"".join(items)}</ul>'
    )

    golden_jsonl = eval_dir / "golden.jsonl"
    golden_html_path = eval_dir / "golden.html"
    golden_section = ""
    if golden_jsonl.exists():
        n = sum(
            1 for line in golden_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
        link = "golden.html" if golden_html_path.exists() else "golden.jsonl"
        golden_section = f"""
  <h2>골든셋</h2>
  <div class="table-wrap">
    <ul class="report-list">
      <li>
        <a class="ts" href="{link}">golden set ({n}건)</a>
        <span class="siblings"><a href="golden.jsonl">JSONL</a></span>
      </li>
    </ul>
  </div>
"""

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>RAGAS 평가 리포트</title>
<style>{_SHARED_CSS}
  body {{ max-width: 900px; }}
</style></head>
<body>
  <h1>RAGAS 평가 리포트</h1>
  <p class="subtitle">RAG 시스템 평가 결과 모음 (최신순)</p>
  {golden_section}
  <h2>실행 결과</h2>
  <div class="table-wrap">
    {results_body}
  </div>
</body></html>
"""
    out = eval_dir / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out


def render_html_report(
    df: Any, meta: dict[str, Any], png_path: Path, out_html: Path,
) -> Path:
    """디자인 리뉴얼: 상단 score 타일 + 차트 + 깔끔한 데이터 테이블."""
    png_b64 = _png_to_base64(png_path)
    summary = meta.get("summary", {})

    # ─── 상단 score 타일 (지표 설명 포함) ──────────────
    tile_html: list[str] = []
    for m in METRICS:
        v = summary.get(m, float("nan"))
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = float("nan")
        color = _METRIC_COLORS.get(m, "#4F46E5")
        display = "—" if np.isnan(fv) else f"{fv:.3f}"
        pct = 0 if np.isnan(fv) else int(round(fv * 100))
        desc = METRIC_DESCRIPTIONS.get(m, "")
        tile_html.append(
            f'<div class="tile" style="--tile-color:{color}">'
            f'  <div class="tile-label">{METRIC_LABELS[m]}</div>'
            f'  <div class="tile-value">{display}</div>'
            f'  <div class="tile-bar"><div class="tile-bar-fill" style="width:{pct}%"></div></div>'
            f'  <div class="tile-desc">{html.escape(desc)}</div>'
            f'</div>'
        )

    # ─── 결과 인사이트 (df 에서 동적 계산) ─────────────
    insights_html = _build_insights(df, summary)

    # ─── 실행 메타 (가로 dl) ─────────────────────────
    def _short(v: Any) -> str:
        s = str(v)
        return s if len(s) <= 60 else s[:57] + "…"

    meta_items = []
    for k in ("timestamp", "git_sha", "n_samples", "model", "embedding_model",
              "top_k", "temperature", "judge_model"):
        if k in meta:
            meta_items.append(
                f'<dt>{html.escape(k)}</dt>'
                f'<dd>{html.escape(_short(meta[k]))}</dd>'
            )

    # ─── 샘플별 데이터 테이블 ────────────────────────
    metric_cols = [m for m in METRICS if m in df.columns]
    has_contexts = "contexts" in df.columns

    rows_html: list[str] = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        q_raw = row.get("question", "") if "question" in df.columns else ""
        a_raw = row.get("answer", "") if "answer" in df.columns else ""
        gt_raw = row.get("ground_truth", "") if "ground_truth" in df.columns else ""

        q_txt, _ = _truncate(q_raw, 200)
        a_txt, _ = _truncate(a_raw, 240)
        gt_txt, _ = _truncate(gt_raw, 200)

        # contexts 는 collapsible
        ctx_html = ""
        if has_contexts:
            ctxs = row.get("contexts") or []
            if isinstance(ctxs, str):
                ctxs = [ctxs]
            if len(ctxs) > 0:
                ctx_items = "".join(
                    f"<li>{html.escape(_truncate(c, 200)[0])}</li>"
                    for c in ctxs[:5]
                )
                ctx_html = (
                    f'<details class="contexts"><summary>contexts ({len(ctxs)})</summary>'
                    f'<ol>{ctx_items}</ol></details>'
                )

        score_cells = []
        for m in metric_cols:
            v = row[m]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            cls = _score_class(fv)
            disp = "—" if np.isnan(fv) else f"{fv:.3f}"
            score_cells.append(f'<td class="score {cls}">{disp}</td>')

        rows_html.append(
            f'<tr>'
            f'<td class="idx">{i}</td>'
            f'<td class="text"><div class="clamp">{html.escape(q_txt)}</div></td>'
            f'<td class="text"><div class="clamp">{html.escape(a_txt)}</div>{ctx_html}</td>'
            f'<td class="text"><div class="clamp">{html.escape(gt_txt)}</div></td>'
            f'{"".join(score_cells)}'
            f'</tr>'
        )

    header_score_cells = "".join(
        f'<th style="text-align:right">{METRIC_LABELS[m]}</th>' for m in metric_cols
    )

    ts = html.escape(str(meta.get("timestamp", "")))
    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>RAGAS Eval — {ts}</title>
<style>{_SHARED_CSS}</style>
</head>
<body>
  <h1>RAGAS 평가 리포트</h1>
  <p class="subtitle">생성: <code>{ts}</code> · 샘플 {meta.get('n_samples', '?')}건 ·
     <a href="./" style="color:var(--primary);text-decoration:none">← 리포트 목록</a></p>

  <section class="tile-grid">
    {''.join(tile_html)}
  </section>

  {insights_html}

  <h2>실행 정보</h2>
  <section class="card">
    <dl class="meta-grid">{''.join(meta_items)}</dl>
  </section>

  <h2>점수 차트</h2>
  <section class="card" style="padding: 12px">
    <img class="chart" src="data:image/png;base64,{png_b64}" alt="RAGAS charts">
  </section>

  <h2>샘플별 데이터</h2>
  <p class="copy-hint">표 영역 드래그 → Ctrl+C 로 엑셀/시트 붙여넣기 가능. CSV 다운로드는
     <a href="{ts and 'ragas_' + ts + '.csv' or '#'}" style="color:var(--primary)">여기</a>.</p>
  <div class="table-wrap">
    <table class="data">
      <thead>
        <tr>
          <th>#</th>
          <th>Question</th>
          <th>Answer</th>
          <th>Ground Truth</th>
          {header_score_cells}
        </tr>
      </thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")
    return out_html
