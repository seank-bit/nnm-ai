# Smoke Test & 200건 정확도 표본 검증 절차

본 문서는 plan Task 29의 수동 실행 절차입니다. subagent로 자동화할 수 없는 실제 환경 검증입니다.

## 사전 준비

1. `.env` 작성:
   ```
   NNM_DB_URL=postgresql+asyncpg://<user>:<pass>@<rds-endpoint>:5432/<db>
   NNM_S3_BUCKET=<your-bucket>
   AWS_ACCESS_KEY_ID=<key>
   AWS_SECRET_ACCESS_KEY=<secret>
   NNM_PUBLICATION_CSV=/app/data/publications.csv
   NNM_PUBLICATION_CSV_ENCODING=cp949
   ```

2. AWS RDS PostgreSQL에 vector 확장 사전 설치 확인
3. publications 매핑 CSV (`publications_*.csv`) 를 repo root 에 둠 — docker-compose
   가 `/app/data/publications.csv` 로 마운트
4. **(prod 만)** 임베딩 모델 사전 다운로드:
   ```bash
   ./scripts/download_models.sh   # ./models/bge-m3/ 에 ~2.2GB
   ```
5. Docker daemon 실행 확인

## 절차

### 1. 컨테이너 기동
```bash
docker compose up app
```
기대: alembic upgrade head 후 0007 도달, `Uvicorn running on http://0.0.0.0:80`

### 2. 헬스/뷰어 확인
```bash
curl http://localhost:8000/healthz
curl -I http://localhost:8000/viewer/
```
기대: 200 OK

### 3. 5건 표본 백필
```bash
docker compose run --rm app \
    python -m nnm.workers.backfill_cli ingest --prefix papers/ --limit 5
```
기대: embedder.loaded, pdf.extracted×5, backfill.ok×5

### 4. 데이터 확인
브라우저로 http://localhost:8000/viewer/ 접속:
- papers ≥ 5
- chunks > 0
- 매핑률 표시 확인

### 5. 200건 정확도 검증
```bash
docker compose run --rm app \
    python -m nnm.workers.backfill_cli ingest --prefix papers/ --limit 200 \
    --job-name "validation-sample"
```

검토 포인트 (/viewer/papers에서 무작위 클릭):
- heading hierarchy 다양성 (Introduction/Methods/...)
- 표/수식 끊김 여부
- 한국어 청크 정상 추출
- 매핑률 90%+ 목표

### 6. 인덱스 적용 확인
```bash
docker compose run --rm app alembic current
```
기대: 0007 (head)

## 검증 결과 기록

| 항목 | 목표 | 실측 | 비고 |
|---|---|---|---|
| 200건 처리 시간 | TBD | | |
| 매핑률 (external_id) | ≥90% | | |
| heading 인식률 | ≥95% | | |
| 표 추출 품질 | 육안 OK | | |
| 한국어 청크 정상 | 육안 OK | | |

이후 본격 10만 건 ingest 진행 여부를 사용자와 협의.
