# nnm-ai — 학술 논문 임베딩 파이프라인 설계 문서

- **작성일**: 2026-05-19
- **상태**: 합의 완료, 구현 계획(`writing-plans`) 전 단계
- **소유**: Sean (seank@hakjisa.co.kr)
- **범위**: AWS S3에 적재된 학술 논문 PDF(~10만 건)를 추출·청킹·임베딩하여 AWS RDS PostgreSQL + `vector` 확장에 적재. FastAPI 기반 + Docker 실행. 검색 API는 본 단계 범위 외(추후 확장).

---

## 1. 목표와 범위

### 1.1 목표
- S3 PDF → opendataloader-pdf로 JSON/Markdown 추출
- bge-m3 임베딩 기준 청킹(섹션 인식, 헤더 프리픽스, references 제외)
- AWS RDS PostgreSQL + `vector` 확장에 적재 (기존 데이터 wipe 후 재적재)
- 데이터 확인용 read-only 뷰어 라우터 제공
- 도커로 실행, 로컬 CPU·EC2 GPU 동일 코드

### 1.2 범위
- 포함: 백필 CLI(추출→청킹→임베딩→적재), 뷰어 라우터, 헬스/메트릭, 운영 DB(`publication_files`) 매핑 조회
- 제외: 외부 ingest API, 검색 API(이후 확장 예정), reranker, 운영 알람

### 1.3 운영 모드
- 1회성 백필 + 확장 가능 구조 (워커 추상화 두되 CLI 진입점 우선)

---

## 2. 핵심 결정 요약

| 항목 | 결정 |
|---|---|
| 청크 크기 | 512 tokens / 64 token overlap (~12%) |
| 청킹 방식 | opendataloader JSON `heading level` → section split → recursive → bge-m3 tokenizer 토큰 검증 |
| 헤더 프리픽스 | `"[Methods] ..."` 형태로 청크 본문 앞에 섹션명 prepend |
| 참고문헌 | 별도 테이블 저장, 임베딩 풀에서 제외 |
| 임베딩 모델 | BAAI/bge-m3 |
| 임베딩 실행 | **in-process** (별도 추론 서버 X) `LocalEmbedder` 단일 구현 |
| 임베딩 환경 | 로컬 CPU(테스트) / EC2 g5.xlarge spot Seoul(운영) — 동일 코드, 환경변수만 분기 |
| 벡터 종류 | dense + sparse는 DB에, ColBERT는 로컬 디렉토리(`var/colbert/`)에 numpy 저장 |
| 데이터베이스 | AWS RDS PostgreSQL + `vector` 확장 (사전 설치됨, RDS 직접 사용) |
| 운영 DB 매핑 | 같은 RDS database 내 `publication_files.s3_key` 조회 → `publication_id`(UUID)를 `papers.external_id`에 저장. 동시에 `publications.title`을 `papers.title`로 우선 채움 |
| 청크 임베딩 prefix | `"[논문 제목] [Methods] 본문..."` — 동일 섹션이라도 어느 논문 소속인지 임베딩 공간에서 분리 |
| 마이그레이션 | Alembic (nnm 전용 테이블만 관리, 운영 테이블은 read-only) |
| 뷰어 | Jinja2 + Tailwind CDN + 최소 HTMX |
| PDF 메타데이터 출처 | PDF 본문에서만 추출 |
| 패키지 매니저 | uv |
| 포매팅·린트 | Ruff (line 100) |
| 타입 체크 | mypy strict |
| 비동기 | 모든 IO async (`aioboto3`, `SQLAlchemy 2.0 async`, `httpx`) |
| 로깅 | structlog JSON + correlation_id |
| CLI | Typer |
| 컨테이너 | 단일 이미지 + `--build-arg DEVICE=cpu|gpu` 분기 |
| 로컬 DB 운용 | 운영 RDS 직접 사용 (같은 database) — `reset_db`는 `--force` 가드 |

---

## 3. 시스템 아키텍처

```
┌───────────────────────────────────────────────────────────┐
│                       AWS S3 (논문 PDF)                    │
└──────────────────────────┬────────────────────────────────┘
                           │ aioboto3 stream
                           ▼
┌───────────────────────────────────────────────────────────┐
│   backfill CLI (nnm.workers.backfill_cli)                 │
│                                                            │
│   ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│   │ PdfExtractor │ →  │   Chunker    │ →  │  Embedder  │ │
│   │ opendataloader│   │ section-aware│    │ LocalEmbed │ │
│   │ -pdf JSON+MD │    │ 512/64       │    │ (in-proc)  │ │
│   └──────┬───────┘    └──────┬───────┘    └─────┬──────┘ │
│          │ var/extracted/     │                  │        │
│          │                    │              var/colbert/ │
│          ▼                    ▼                  ▼        │
│   ┌─────────────────────────────────────────────────────┐│
│   │   Repository (SQLAlchemy 2.0 async + asyncpg)       ││
│   │   - papers, chunks, chunk_embeddings (write)        ││
│   │   - publication_files (read-only join)              ││
│   └──────────────────────────┬──────────────────────────┘│
└──────────────────────────────┼──────────────────────────────┘
                               │
                               ▼
          ┌────────────────────────────────────────┐
          │   AWS RDS PostgreSQL + vector ext       │
          │   [nnm 관리]                            │
          │     papers, chunks, chunk_embeddings,   │
          │     paper_references, ingest_jobs,      │
          │     ingest_job_items                    │
          │   [운영 관리 - read-only]                │
          │     publication_files (s3_key, pub_id)  │
          │     publications (id, title)            │
          └─────────────────┬──────────────────────┘
                            ▲
                            │
┌───────────────────────────┴────────────────────────────────┐
│   FastAPI app (nnm.main)                                    │
│   - /viewer/  (Jinja2 read-only 대시보드)                    │
│   - /healthz, /metrics                                       │
│   - 검색 API는 추후 확장                                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 단방향 의존
`domain → services → api / workers` (역방향 import 금지). `infra` 어댑터는 `services`에서만 호출.

---

## 4. 데이터 파이프라인

### 4.1 처리 단계
1. **목록화**: S3 `prefix` 아래 객체를 paginate, 신규/실패 키만 `ingest_job_items`에 enqueue
2. **다운로드 + 해시**: 스트림 단위 다운로드, sha256 계산, 기존 `papers.file_hash`와 중복 시 skip
3. **운영 DB 매핑 조회**:
   ```sql
   SELECT pf.publication_id, p.title
   FROM publication_files pf
   JOIN publications p ON p.id = pf.publication_id
   WHERE pf.s3_key = :s3_key
   ```
   - 매치 시 `papers.external_id`와 `papers.title`에 저장
   - 미매치 시 WARNING 로그, `external_id NULL`로 진행 (백필 차단 X)
4. **추출**: opendataloader-pdf 호출 → JSON과 Markdown 동시 생성 → `var/extracted/{paper_hash}.json|.md`
5. **메타 파싱(보완)**: 3에서 title을 확보하지 못했으면 JSON heading_level=1 또는 metadata.title 폴백. 저자/abstract/언어/페이지 수는 항상 PDF에서 추출
6. **청킹**: 섹션 인식 → recursive 분할 → bge-m3 tokenizer 검증
7. **임베딩**: 청크 배치 단위로 `LocalEmbedder.embed()` (dense + sparse + ColBERT)
8. **저장**: dense/sparse는 `chunk_embeddings`에 INSERT, ColBERT는 `var/colbert/{chunk_id}.npy` 저장 후 path만 컬럼 기록
9. **체크포인트**: `ingest_job_items.status = 'embedded'` 업데이트, 트랜잭션 단위 1~5k 청크

### 4.2 opendataloader-pdf 옵션
```
--format json,markdown
--reading-order xycut
--use-struct-tree
--threads <core수>
--table-method cluster
--image-output external
# hybrid/OCR/--enrich-formula 비활성 (처리량 보존)
```

본격 10만 건 처리 전, **200개 무작위 표본 검증** 단계를 파이프라인에 포함:
- heading hierarchy 보존
- table-method cluster 표 재구성 품질
- 2단 references reading order
- 스캔 한국어가 있으면 별도 표본 검증

### 4.3 청킹 알고리즘 의사코드
```
def chunk(paper_json):
    sections = split_by_heading(paper_json)        # heading_level == 1,2 기준
    sections = drop_section(sections, name="References")
    paper_title = paper.title or ""               # 운영 DB publications.title 우선, 폴백은 PDF
    for section in sections:
        section_name = section.heading
        text = section.text                         # 표/수식은 atomic placeholder로 보존
        for sub in recursive_char_split(text, target_tokens=512):
            tokens = bge_m3_tokenizer.encode(sub)
            if len(tokens) > 512:
                sub = retokenize_and_trim(sub, 512)
            prefix = f"[{paper_title}] [{section_name}]" if paper_title else f"[{section_name}]"
            yield Chunk(
                section=section_name,
                text=sub,
                text_for_embed=f"{prefix} {sub}".strip(),
                token_count=len(tokens),
                page_from=..., page_to=...,
            )
```

### 4.4 임베딩 호출
- Batch size: 로컬 4, EC2 32
- FP16: 로컬 false, EC2 true
- Output: `return_dense=True, return_sparse=True, return_colbert_vecs=True`

---

## 5. 데이터 모델 (DDL)

### 5.1 `papers`
```sql
CREATE TABLE papers (
    id              BIGSERIAL PRIMARY KEY,
    s3_key          TEXT        NOT NULL UNIQUE,
    file_hash       CHAR(64)    NOT NULL,
    external_id     UUID,                          -- publication_files.publication_id
    title           TEXT,
    authors         TEXT[],
    abstract        TEXT,
    venue           TEXT,
    published_year  SMALLINT,
    language        VARCHAR(8),
    page_count      INT,
    raw_json_path   TEXT,
    raw_md_path     TEXT,
    status          VARCHAR(16) NOT NULL DEFAULT 'pending',
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX papers_status_idx    ON papers (status);
CREATE INDEX papers_year_idx      ON papers (published_year);
CREATE INDEX papers_language_idx  ON papers (language);
CREATE UNIQUE INDEX papers_external_id_idx
  ON papers (external_id) WHERE external_id IS NOT NULL;
```

### 5.2 `chunks`
```sql
CREATE TABLE chunks (
    id              BIGSERIAL PRIMARY KEY,
    paper_id        BIGINT      NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    seq             INT         NOT NULL,
    section         TEXT,
    section_level   SMALLINT,
    page_from       INT,
    page_to         INT,
    token_count     INT         NOT NULL,
    char_count      INT         NOT NULL,
    text            TEXT        NOT NULL,
    text_for_embed  TEXT        NOT NULL,
    language        VARCHAR(8),
    UNIQUE (paper_id, seq)
);
CREATE INDEX chunks_paper_idx    ON chunks (paper_id);
CREATE INDEX chunks_language_idx ON chunks (language);
```

### 5.3 `chunk_embeddings`
```sql
CREATE TABLE chunk_embeddings (
    chunk_id       BIGINT       NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model_name     VARCHAR(64)  NOT NULL,
    model_version  VARCHAR(32)  NOT NULL,
    dense          VECTOR(1024) NOT NULL,
    sparse         JSONB,
    colbert_path   TEXT,                       -- var/colbert/{chunk_id}.npy
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model_name, model_version)
);
```

### 5.4 `paper_references`
```sql
CREATE TABLE paper_references (
    id          BIGSERIAL PRIMARY KEY,
    paper_id    BIGINT      NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    seq         INT         NOT NULL,
    raw_text    TEXT        NOT NULL,
    UNIQUE (paper_id, seq)
);
```

### 5.5 `ingest_jobs`, `ingest_job_items`
```sql
CREATE TABLE ingest_jobs (
    id             BIGSERIAL PRIMARY KEY,
    job_name       TEXT        NOT NULL,
    s3_prefix      TEXT        NOT NULL,
    total_files    INT,
    processed      INT         NOT NULL DEFAULT 0,
    failed         INT         NOT NULL DEFAULT 0,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ
);

CREATE TABLE ingest_job_items (
    job_id      BIGINT      NOT NULL REFERENCES ingest_jobs(id) ON DELETE CASCADE,
    s3_key      TEXT        NOT NULL,
    paper_id    BIGINT      REFERENCES papers(id),
    status      VARCHAR(16) NOT NULL DEFAULT 'pending',
    error       TEXT,
    attempts    SMALLINT    NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, s3_key)
);
CREATE INDEX ingest_items_status_idx ON ingest_job_items (job_id, status);
```

### 5.6 운영 테이블 (read-only)
- `publication_files` — 운영 시스템이 관리. nnm 마이그레이션에서 **만들지 않음**
  - 사용 컬럼: `s3_key`, `publication_id (UUID)`
- `publications` — 운영 시스템이 관리. nnm 마이그레이션에서 **만들지 않음**
  - 사용 컬럼: `id (UUID)`, `title (TEXT)`
- nnm 코드는 조인 후 SELECT만 수행 (§4.1 step 3 참조)
- SQLAlchemy 모델은 read-only 명시적 정의 — 운영 시스템이 스키마를 변경하면 nnm 쪽에서 동기 책임

### 5.7 인덱스 (백필 완료 후 일괄 생성)
```sql
CREATE INDEX chunk_emb_dense_hnsw
ON chunk_embeddings
USING hnsw (dense vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX chunk_emb_sparse_gin
ON chunk_embeddings
USING gin (sparse jsonb_path_ops);

CREATE INDEX chunks_text_gin
ON chunks
USING gin (to_tsvector('simple', text));
```

### 5.8 Alembic 마이그레이션 순서
1. `0001_install_vector_extension.py` — `CREATE EXTENSION IF NOT EXISTS vector`
2. `0002_create_papers.py`
3. `0003_create_chunks.py`
4. `0004_create_chunk_embeddings.py`
5. `0005_create_paper_references.py`
6. `0006_create_ingest_jobs.py`
7. `0007_create_indexes.py`

### 5.9 `reset_db` 동작
- nnm 관리 테이블만 drop. **운영 테이블(`publication_files`)은 절대 건드리지 않음**
- 구현: `DROP TABLE IF EXISTS papers, chunks, chunk_embeddings, paper_references, ingest_jobs, ingest_job_items, alembic_version CASCADE`
- 이어서 `alembic upgrade head`
- 가드: 어느 환경이든 `--force` 플래그 필요 (로컬·운영이 같은 DB이므로 보수적으로)

---

## 6. 코드 컨벤션

### 6.1 표준
- Python **3.12+**
- 패키지 매니저 **uv**, `uv.lock` 커밋
- **Ruff** (line 100) — `E, W, F, I, B, C4, UP, SIM, ARG, PIE, RUF, ASYNC, TID, ANN, C901`
- **mypy strict**, `from __future__ import annotations` 모든 파일
- **pydantic v2 / pydantic-settings v2**
- 비동기 IO: `aioboto3`, `SQLAlchemy 2.0 async`, `asyncpg`, `httpx.AsyncClient`
- 동기 SDK 사용 금지 (`boto3`, `requests`, sync session 등)
- **ecc:python-patterns** 보조 기준으로 적용

### 6.2 디렉토리 (src layout)
```
nnm-ai/
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── Dockerfile                  # single image, build-arg DEVICE
├── alembic.ini
├── .env.example
├── .dockerignore
├── docs/superpowers/specs/
├── src/nnm/
│   ├── main.py
│   ├── config.py
│   ├── logging.py
│   ├── errors.py
│   ├── db/
│   │   ├── session.py
│   │   ├── models.py          # publication_files 포함 (read-only)
│   │   └── migrations/
│   ├── domain/
│   │   ├── paper.py
│   │   ├── chunk.py
│   │   └── embedding.py
│   ├── services/
│   │   ├── s3_loader.py
│   │   ├── pdf_extractor.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── publication_mapper.py    # 운영 매핑 조회
│   │   ├── backfill.py
│   │   └── maintenance.py
│   ├── api/
│   │   ├── deps.py
│   │   └── viewer/
│   │       ├── router.py
│   │       └── templates/
│   ├── workers/
│   │   └── backfill_cli.py
│   └── infra/
│       ├── s3.py
│       └── local_embedder.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
└── var/
    ├── extracted/
    └── colbert/
```

### 6.3 단방향 의존
`domain` 패키지는 SQLAlchemy/FastAPI/외부 SDK import 금지. `services`에서 어댑터를 주입.

### 6.4 에러 처리
- `NnmError` 최상위, 도메인별 서브클래스 (`PaperNotFound`, `PdfExtractionError`, `EmbeddingFailure`, `S3FetchError`, `PublicationMappingMissed`)
- FastAPI exception handler에서 HTTP 상태 매핑
- 외부 호출에 `tenacity` 재시도 (exponential backoff + jitter)
- silent failure 금지 — `except: pass` 금지
- 매핑 미스(`PublicationMappingMissed`)는 예외 던지지 말고 WARNING + `external_id = NULL`로 진행

### 6.5 로깅
- structlog JSON(운영) / 컬러 콘솔(로컬)
- 요청별 `correlation_id` 미들웨어 자동 주입
- prometheus_client 메트릭 `/metrics` 노출

### 6.6 테스트
- pytest + pytest-asyncio + pytest-cov
- integration test는 실제 Postgres + vector 확장 사용 (mock 금지)
- coverage 80%+, domain 95%+

---

## 7. FastAPI 골격

### 7.1 진입점 (`main.py`)
- `create_app()` 팩토리
- `lifespan` 훅에서 `LocalEmbedder.load()` 사전 호출
- `correlation_id` 미들웨어
- 라우터 등록: `/viewer/...`, `/healthz`, `/metrics`

### 7.2 의존성 주입 (`api/deps.py`)
- `Annotated[..., Depends(...)]` 타입 별칭으로 라우터 시그니처 간결화
- `SettingsDep`, `DbDep`, `EmbedderDep`, `S3Dep`

### 7.3 뷰어 라우터 페이지
| 경로 | 내용 |
|---|---|
| `/viewer/` | 대시보드: 논문/청크 카운트, 상태별, 최근 ingest_jobs, **매핑 성공률** |
| `/viewer/papers` | 페이지네이션 + 언어/상태/연도 필터, `external_id` 표시 |
| `/viewer/papers/{id}` | 메타 + 청크 목록 + raw 경로 + 매핑된 `publication_id` |
| `/viewer/chunks/{id}` | 청크 본문, embedding 메타 |
| `/viewer/jobs` | ingest_jobs 진행률, 최근 실패 |

스택: Jinja2 + Tailwind CDN + 최소 HTMX.

### 7.4 CLI (`workers/backfill_cli.py`, Typer)
```
nnm ingest --prefix papers/ --limit 100   # 백필 (옵션: 일부만)
nnm reset-db --force                      # nnm 테이블 전체 wipe + 재마이그레이션
```

---

## 8. 임베딩 실행

### 8.1 `LocalEmbedder` (in-process)
- `BGEM3FlagModel`을 lazy-load
- FastAPI lifespan과 CLI 진입에서 `load()` 호출
- `asyncio.to_thread()`로 동기 encode를 비차단 wrap

### 8.2 환경 분기 (코드 동일, 환경변수만 다름)

| 변수 | 로컬 | EC2 |
|---|---|---|
| `NNM_EMBEDDING_DEVICE` | `cpu` | `cuda` |
| `NNM_EMBEDDING_FP16` | `false` | `true` |
| `NNM_EMBEDDING_BATCH_SIZE` | `4` | `32` |
| `NNM_EMBEDDING_MAX_TOKENS` | `8192` | `8192` |

### 8.3 EC2 운영 권장 인프라
- 인스턴스: **g5.xlarge spot ap-northeast-2** (Seoul), on-demand 폴백
- AMI: AWS Deep Learning AMI (CUDA 12.x)
- 백필 예상: 10M chunks × dense+sparse+ColBERT 산출 ≈ 28h × $0.37 ≈ **~$10**
- 운영 트래픽(1k chunks/day): 상시 가동 X — AWS Batch spot 또는 CPU+ONNX 컨테이너

### 8.4 체크포인팅
- (paper_id, chunk_id) → vectors를 batch 1~5k 단위로 Postgres에 즉시 커밋
- spot 중단 시 손실 최소화
- 첫 1k 청크는 워밍업 (처리량 측정 제외)

---

## 9. Docker 구성

### 9.1 단일 Dockerfile, build-arg 분기
```dockerfile
ARG DEVICE=cpu
FROM python:3.12-slim AS base
# openjdk-17 (opendataloader-pdf JVM), libpq, uv 설치
FROM base AS build-cpu
RUN uv pip install --system -e ".[cpu]"
FROM base AS build-gpu
RUN uv pip install --system --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.5" && uv pip install --system -e ".[gpu]"
FROM build-${DEVICE} AS final
CMD ["uvicorn", "nnm.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 9.2 docker-compose.yml (RDS 직접 연결, postgres 서비스 없음)
- `app` 서비스만 존재
- `NNM_DB_URL`은 RDS 엔드포인트 (운영 DB와 동일)
- `var/` 디렉토리 volume mount
- `command`: alembic upgrade → uvicorn `--reload`

### 9.3 실행 흐름
```bash
cp .env.example .env                                          # 값 입력
docker compose up app                                         # 앱 기동 + 자동 마이그레이션

docker compose run --rm app \
    python -m nnm.workers.backfill_cli ingest --prefix papers/

docker compose run --rm app \
    python -m nnm.workers.backfill_cli reset-db --force        # 위험

open http://localhost:8000/viewer/
```

---

## 10. 결정된 리스크 및 가드레일

| 리스크 | 가드레일 |
|---|---|
| `reset_db` 우발 실행 → 운영 데이터 손실 | `--force` 플래그 필수. drop 대상은 nnm 관리 테이블만, 운영 테이블은 절대 미터치. 로컬·운영이 같은 DB이므로 보수적 |
| `publication_files`/`publications` 스키마 변경에 의한 매핑 실패 | 매핑 미스는 예외 X, WARNING + `external_id NULL`로 계속 진행. 매핑 성공률은 뷰어 대시보드에 노출 |
| 논문 제목 미확보 (운영 DB·PDF 모두 실패) | `text_for_embed`에서 title prefix 생략, `[Section] body` 포맷으로 degraded. title 누락률은 뷰어 대시보드에 노출 |
| opendataloader-pdf 정확도 | 본격 처리 전 200개 무작위 표본 검증 단계 의무화 |
| 스캔 한국어 PDF 처리 품질 | 해당 표본 별도 검증, 필요 시 hybrid/OCR로 폴백 |
| ColBERT 저장 폭증 | DB 미적재. 로컬 디렉토리에 npy로만 보관, 경로만 컬럼 기록 |
| spot 인스턴스 중단 | 1~5k 청크 단위 체크포인트로 손실 최소화 |
| seq 8192 직접 임베딩 OOM | 청킹 후 임베딩 — raw 논문 직접 입력 금지 |
| 임베딩 처리량 측정 오차 | 첫 1k 청크는 워밍업 제외 |
| 한국어 토큰 인플레이션 | 청크 길이 검증은 bge-m3 tokenizer로 토큰 카운트 (char 추정 금지) |
| HF Inference Provider의 CPU 한계 | 사용 안 함 — 자체 호스팅 g5.xlarge로 결정 |

---

## 11. 본 단계 범위에서 제외된 항목 (추후 확장)

- 검색 API (`/api/v1/search`) — 인덱스는 사전 구축
- 외부 ingest API — CLI 백필로 대체
- ColBERT 활용한 reranker
- 운영 모니터링·알람 (Prometheus 메트릭은 노출하되 알람 룰 미정의)
- AWS Secrets Manager / SSM 시크릿 주입 (현재는 .env)
- 워커 큐 시스템 (Celery/arq) — 인터페이스만 추상화, 도입은 향후

---

## 12. 참고 자료

- BGE-M3 모델 카드: <https://huggingface.co/BAAI/bge-m3>
- BGE-M3 chunk size 권장(BAAI Shitao): <https://huggingface.co/BAAI/bge-m3/discussions/59>
- BGE-M3 paper: <https://arxiv.org/html/2402.03216v3>
- opendataloader-pdf: <https://github.com/opendataloader-project/opendataloader-pdf>
- opendataloader CLI 옵션: <https://opendataloader.org/docs/reference/cli-options>
- HF Inference Providers 가격: <https://huggingface.co/docs/inference-providers/pricing>
- pgvector: <https://github.com/pgvector/pgvector>
- Infinity (참고 — 본 설계에서는 미사용): <https://github.com/michaelfeil/infinity>
