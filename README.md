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
open http://localhost/viewer/
```

## EC2 GPU (운영계)
```bash
# 1) 이미지 빌드
docker compose -f docker-compose.yml -f docker-compose.prod.yml build

# 2) 임베딩 모델 1회 다운로드 (HF → ./models/bge-m3/, ~2.2GB)
#    prod compose 는 HF_HUB_OFFLINE=1 이므로 미리 받아두어야 함.
./scripts/download_models.sh

# 3) publications 매핑 CSV 를 repo root 에 둠 (NNM_PUBLICATION_CSV 가 가리키는 경로)
ls publications_*.csv

# 4) prod 컨테이너 기동
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

설계: `docs/superpowers/specs/2026-05-19-nnm-ai-design.md`
