# RAGAS 평가 골든 데이터셋

골든셋 파일 위치: **`var/eval/golden.jsonl`** (호스트 ↔ 컨테이너 양쪽 모두 보임).
이 디렉토리(`tests/eval/`)는 스키마 문서/예시 보관용입니다.

한 줄당 한 평가 항목 JSONL.

## 스키마

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `question` | str | ✅ | 사용자가 RAG 에 입력할 질문 |
| `ground_truth` | str | ✅ | 사람이 작성한 정답 (논문 근거 기반) |
| `ground_truth_chunk_ids` | list[int] | ❌ | 정답 근거 chunk_id (보조 retrieval 측정용) |

## 권장 규모

- 최소 30건, 분야별 골고루
- 어려운 사례(다중 근거, 부정문, 비교 질문) 30% 이상 포함

## 작성 팁

1. 실제 DB 에 적재된 논문 범위 안에서 질문 생성
2. ground_truth 는 한 문장 ~ 세 문장이 적정 (너무 길면 context_recall 측정이 노이지해짐)
3. `ground_truth_chunk_ids` 는 PSQL 로 직접 확인:
   ```sql
   SELECT id, paper_id, section, left(text, 100)
   FROM chunks
   WHERE paper_id = <X> ORDER BY seq;
   ```

## 자동 생성 (출발점)

손으로 0부터 작성하기 부담스러우면 generator 사용:

```bash
python -m nnm.eval.gen_golden --n 30           # 기본 출력: var/eval/golden.jsonl
```

코퍼스에서 chunk 를 샘플링한 뒤 Groq LLM 으로 (질문, 정답) 쌍을 만들어
`ground_truth_chunk_ids` 까지 자동으로 채웁니다. **검수 필수**.
