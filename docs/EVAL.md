# RAG 평가 (RAGAS)

본 프로젝트의 RAG 파이프라인(pgvector + bge-m3 + Groq llama-3.3-70b)을
[RAGAS](https://docs.ragas.io) 의 네 가지 표준 지표로 측정합니다.

## 측정 지표

| 지표 | 의미 | 입력 |
|---|---|---|
| **Faithfulness** | 답변이 컨텍스트에 충실한가 (환각 여부) | context, answer |
| **Context Recall** | ground_truth 중 context 로 회수된 비율 | ground_truth, context |
| **Context Precision** | 관련 context 가 상위 순위에 잘 검색됐는가 | ground_truth, context (순위) |
| **Answer Relevance** | 답변이 질문과 얼마나 관련 있는가 | question, answer |

## 1. 설치

```bash
# 평가용 의존성만 추가 설치 (런타임 이미지엔 들어가지 않음)
pip install -e ".[eval]"
```

`ragas`, `datasets`, `langchain-openai`, `langchain-huggingface`,
`sentence-transformers` 등이 설치됩니다.

## 2. 환경 변수

| 변수 | 용도 |
|---|---|
| `NNM_GROQ_API_KEY` | RAG 답변 생성 + (기본) RAGAS judge |
| `OPENAI_API_KEY` | (선택) judge LLM 을 OpenAI 로 분리하고 싶을 때 |
| `OPENAI_BASE_URL` | (선택) OPENAI_API_KEY 와 함께 base URL 재정의 |
| `RAGAS_JUDGE_MODEL` | (선택) judge 모델 명시 — 미지정 시 `NNM_GROQ_MODEL` 사용 |

> Groq 의 rate limit / 한국어 평가 신뢰도가 문제일 경우 judge 만 OpenAI
> (예: `gpt-4o-mini`) 로 분리하는 것을 권장.

## 3. 골든셋 작성

골든셋 파일 경로: **`var/eval/golden.jsonl`** (volume-mounted — 호스트와 컨테이너 양쪽에서 보임).
한 줄당 한 항목:

```jsonl
{"question": "...", "ground_truth": "...", "ground_truth_chunk_ids": [12, 34]}
```

자세한 스키마는 [tests/eval/README.md](../tests/eval/README.md) 참고.

### 3-1. Synthetic 자동 생성 (권장 출발점)

DB 에서 chunk 를 샘플링하고 Groq LLM 으로 (질문, 정답) 쌍을 자동 생성:

```bash
python -m nnm.eval.gen_golden --n 30           # → var/eval/golden.jsonl
```

생성된 파일은 **반드시 사람이 검수**하세요:
- 질문이 너무 일반적이거나 단답형이면 삭제/수정
- ground_truth 가 본문과 다르면 수정
- `ground_truth_chunk_ids` 는 자동으로 채워집니다 (출처 chunk_id)

옵션:
- `--n 30` 생성할 항목 수
- `--min-chars 400` chunk 최소 길이 (너무 짧으면 답할거리가 없음)
- `--seed 7` 동일 결과 재현용 시드

## 4. 실행

```bash
python -m nnm.eval.run_ragas \
    --golden tests/eval/golden.jsonl \
    --out var/eval

# 일부만 빠르게 확인
python -m nnm.eval.run_ragas --golden tests/eval/golden.jsonl --limit 5

# CI: 임계값 미달 시 exit 1
python -m nnm.eval.run_ragas --enforce
```

결과:
- `var/eval/ragas_<timestamp>.csv` — 샘플별 점수 (질문/답변/지표 전체) — **엑셀 직접 열기 가능**
- `var/eval/ragas_<timestamp>.json` — git_sha, 모델, top_k, 평균 지표 요약
- `var/eval/ragas_<timestamp>.png` — 차트 3종 (막대 / 레이더 / 히트맵)
- `var/eval/ragas_<timestamp>.html` — 차트 + 메타 + 엑셀 복붙용 데이터 테이블

### 4-1. 브라우저로 보기

FastAPI 앱이 `/eval/` 경로에 결과 폴더를 정적 mount 합니다
([src/nnm/main.py](../src/nnm/main.py)). 앱 실행 중이라면:

- `http://<host>:<port>/eval/` — 모든 리포트 목록 (최신순)
- `http://<host>:<port>/eval/ragas_<ts>.html` — 개별 리포트
- `http://<host>:<port>/eval/ragas_<ts>.csv` — CSV 다운로드

`run_ragas` 실행이 끝나면 `index.html` 이 자동 갱신됩니다. 앱 재시작 불필요.

### 4-2. 시각화 구성

`.png` / `.html` 안에 포함되는 차트:

| 차트 | 용도 |
|---|---|
| **막대 (Bar)** | 4개 지표 평균을 한눈에 비교 |
| **레이더 (Radar)** | 4축 균형 — 약한 지표 즉시 식별 |
| **히트맵 (Heatmap)** | 샘플(Q1…Qn) × 지표, 어느 질문이 어디서 깨지는지 진단 |

`.html` 하단 데이터 테이블은 점수 임계값별 색상 (≥0.8 초록 / ≥0.6 노랑 / 그 외 빨강)
및 셀 드래그 → Ctrl+C 로 **엑셀/구글시트에 그대로 붙여넣기** 가능합니다.

## 5. 임계값 (DEFAULT_THRESHOLDS)

[src/nnm/eval/run_ragas.py](../src/nnm/eval/run_ragas.py#L34) 에 정의되어 있으며,
런별로 비교 가능한 회귀 가드입니다. 초기값:

```python
faithfulness:       0.70
context_recall:     0.60
context_precision:  0.60
answer_relevancy:   0.70
```

골든셋 안정화 후 실제 베이스라인 점수에 맞춰 상향 조정하세요.

## 6. 알려진 한계

- **한국어 judge 신뢰도**: RAGAS 프롬프트는 영어 위주. 가능하면 강한 judge
  (gpt-4o-mini 이상) 권장.
- **judge 비용**: 한 샘플당 judge LLM 이 다회 호출됨. 30건 평가에 수백 회.
- **DB 의존**: 평가는 실제 pgvector 검색을 호출하므로 DB 가 실행 중이고
  `ChunkEmbedding` 이 적재돼 있어야 함.
