# nnm-ai

학술 논문 임베딩 파이프라인. PDF → opendataloader-pdf → bge-m3 → PostgreSQL+vector.

## 실행
```bash
cp .env.example .env
docker compose up app

# 백필
docker compose run --rm app python -m nnm.workers.backfill_cli ingest --prefix papers/

# 리셋 (위험)
docker compose run --rm app python -m nnm.workers.backfill_cli reset-db --force

# 뷰어
open http://localhost:8000/viewer/
```

## EC2 GPU
```bash
docker build --build-arg DEVICE=gpu -t nnm-ai:gpu .
docker run --gpus all -e NNM_EMBEDDING_DEVICE=cuda -e NNM_EMBEDDING_FP16=true \
  -e NNM_EMBEDDING_BATCH_SIZE=32 -e NNM_DB_URL=... nnm-ai:gpu \
  python -m nnm.workers.backfill_cli ingest --prefix papers/
```

설계: `docs/superpowers/specs/2026-05-19-nnm-ai-design.md`
