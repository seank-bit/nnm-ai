# nnm-ai viewer redesign — design spec

- **Date:** 2026-05-20
- **Scope:** `src/nnm/api/viewer/` 전체 (5개 페이지)
- **Goal:** 현재의 평면적 흰색-카드 UI를 Linear/Vercel 스타일의 조용한 미니멀 톤으로 통일하고, 메트릭 카드에 7일 추이 sparkline 을 추가한다.

## 1. 비주얼 시스템

### 컬러
- 베이스: `zinc` (50/100/200/500/700/900).
- 액센트: `indigo-500` / `indigo-600` (단일).
- 시맨틱: 성공 `emerald-500/600`, 경고 `amber-500`, 실패 `rose-500/600`.
- 배경: 페이지 `#ffffff`, 카드 표면도 `#ffffff` + `border border-zinc-200`.

### 타이포그래피
- 본문 폰트: Inter (400/500/600/700), Google Fonts CDN.
- 모노 폰트: JetBrains Mono → fallback `ui-monospace`.
- 본문 14px / `text-zinc-700`, 큰 수치 `text-3xl font-semibold tracking-tight text-zinc-900`.
- 보조 텍스트 `text-sm text-zinc-500`.

### 모양/간격
- 모서리: 카드 `rounded-lg` (8px), 인풋 `rounded-md` (6px), 칩/점 `rounded-full`.
- 그림자: `shadow-sm` 최소만 — 경계는 1px `border-zinc-200`으로 표현 (Linear/Vercel 핵심).
- 그리드 간격 8px 베이스. 카드 패딩 24px, 섹션 간 40px (`space-y-10`).

### Tailwind config
`base.html` 안에 CDN 스크립트 직후 인라인 config 로드:
```html
<script>
  tailwind.config = {
    theme: {
      extend: {
        fontFamily: {
          sans: ['Inter', 'ui-sans-serif', 'system-ui'],
          mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular'],
        },
      },
    },
  };
</script>
```

## 2. 레이아웃 셸 (`base.html`)

- `<body>`: `bg-white text-zinc-900 font-sans antialiased`.
- 상단 네비: `sticky top-0 z-30 bg-white/80 backdrop-blur border-b border-zinc-200 h-14`.
  - 좌측: 워드마크 `nnm·ai` (`text-sm font-semibold tracking-tight`) + 작은 indigo dot.
  - 우측: `Overview / Papers / Jobs` 링크. 비활성 `text-zinc-500 hover:text-zinc-900`, 활성 `text-zinc-900` + `border-b-2 border-indigo-600 -mb-px`.
- 메인 컨테이너: `max-w-7xl mx-auto px-6 py-10`.
- 푸터 없음.

## 3. Dashboard (`dashboard.html`)

### 헤더
- `<h1 class="text-2xl font-semibold tracking-tight">Overview</h1>`
- `<p class="text-sm text-zinc-500 mt-1">ingest 파이프라인 현황</p>`

### 메트릭 카드 (3 cols → 2 → 1 grid)
각 카드 구조:
```
[label]               ↘ sparkline (64×24)
[big number]
[delta chip] [unit]
```
- 라벨: `text-sm text-zinc-500`.
- 큰 숫자: `text-3xl font-semibold tracking-tight tabular-nums`.
- Delta chip: `inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium`,
  - 양수: `bg-emerald-50 text-emerald-700`,
  - 음수: `bg-rose-50 text-rose-700`,
  - 0/없음: `bg-zinc-100 text-zinc-600`.

카드 3종: Papers (total + 오늘 증가량 + 7일 sparkline) / Chunks (동일) / Mapping rate (% + 1주 변화 pp + 비율 sparkline).

### Sparkline
- 서버측 Jinja 매크로 `sparkline(values, w=64, h=24)`.
- 값 정규화: `min(values), max(values)` 기준으로 0..1 → SVG path `M x0,y0 L x1,y1 ...`.
- 스트로크 1.5px `stroke-indigo-500 fill-none`.
- 값이 전부 동일하거나 < 2개면 가로선만 표시.

### Status 분포
표 → 가로 누적 바로 교체:
```
ready          ████████████████████░░  11,820   95.3%
processing     ▓                          412    3.3%
failed         ▒                          176    1.4%
```
- 바: `h-1.5 rounded-full bg-zinc-100`, 채움은 status 별 색 (ready=indigo-500, processing=amber-500, failed=rose-500, 기타=zinc-400).
- 카드 1개 안에 행으로 나열.

### Recent jobs
- 카드 안에 `<table>`. 헤더 `text-xs uppercase tracking-wide text-zinc-500 border-b border-zinc-200`.
- 행: 높이 44px, 경계 `border-b border-zinc-100`.
- status 점 + `processed/failed` 숫자 `tabular-nums text-right`.

## 4. Papers list (`papers_list.html`)

- 헤더 행: 제목 + 우측 작은 버튼 `+ 필터` 토글 (open 시 필터 row 노출).
- 필터 행: text input `border-zinc-300 focus:ring-indigo-500 rounded-md text-sm`.
- 표:
  - 헤더 sticky (`sticky top-14 bg-white z-10`), 폰트 `text-xs uppercase tracking-wide text-zinc-500`.
  - 행: zebra 없음, 하단 `border-b border-zinc-100`, hover `bg-zinc-50`.
  - `title` 셀: 제목 위 / `external_id`를 그 아래 `font-mono text-xs text-zinc-500` 한 줄로.
  - `status` 셀: 컬러 점 + status 단어 `text-sm text-zinc-600`.
  - `chunks` 셀: `tabular-nums text-right`.
- 페이지네이션: `‹ Page N of T ›` + page size 셀렉터 (20/50/100). 작은 outline 버튼.

## 5. Paper detail (`paper_detail.html`)

- Breadcrumb: `Papers / #{id}` 회색 작게.
- 제목: `text-2xl font-semibold tracking-tight`.
- 메타 라인: `id / external_id / lang / pages / status` chip 형태로 `flex-wrap gap-2`.
- 본문 카드: 2열 grid (`grid-cols-2 gap-x-8 gap-y-3`), key는 `font-mono text-xs text-zinc-500`, value는 `text-sm text-zinc-700 break-all`.
- 청크 표는 Papers list 와 동일 스타일.

## 6. Chunk detail (`chunk_detail.html`)

- Breadcrumb: `Papers / #{paper_id} / Chunk #{id}`.
- 메타 chip: `seq / section / tokens / chars`.
- 코드 블록: `<pre class="bg-zinc-50 border border-zinc-200 rounded-md p-4 text-sm font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap">`.
- 임베딩 섹션: 정의 리스트 (`dl`) — model, version, dense dim, sparse tokens, colbert path. 각 키 `text-xs uppercase tracking-wide text-zinc-500`.

## 7. Jobs (`jobs.html`)

- Papers list 와 동일 표 스타일.
- `failed > 0` 행: 좌측에 `border-l-2 border-rose-500` 보조 표시.
- "최근 실패" 섹션: 같은 카드 패턴, error 컬럼은 `truncate max-w-md` + title 속성으로 풀텍스트.

## 8. 매크로 파일 신설: `_macros.html`

다음 매크로를 정의:
- `metric_card(label, value, delta, delta_kind, spark_values)`
- `sparkline(values, w=64, h=24)`
- `status_dot(status)` — status → 색 매핑된 `<span class="inline-block w-1.5 h-1.5 rounded-full ..." />`
- `pill(text, kind)` — kind ∈ `default | success | warning | danger | info`
- `pagination(page, total_pages, page_size, page_sizes)`

각 매크로는 순수 마크업만 생성 (Jinja `{% macro %}`).

## 9. 라우터 변경 (`router.py`)

`dashboard` 핸들러에 다음을 추가:

- 최근 7일 일자별 카운트 3쿼리:
  ```sql
  SELECT date_trunc('day', created_at) AS d, count(*)
    FROM papers
   WHERE created_at >= now() - interval '7 days'
   GROUP BY 1 ORDER BY 1;
  ```
  동일 패턴을 `chunks`, `chunk_embeddings` 에 대해서도.
- 결과를 7개 길이의 정수 리스트로 정규화 (빠진 날짜는 0 채움) → 템플릿에 `papers_spark`, `chunks_spark`, `mapping_spark` 로 전달.
- "오늘 증가" 델타 = `spark[-1]` (또는 마지막 24시간 별도 쿼리 — 단순 단일 SELECT count로).
- Mapping rate sparkline 은 `papers WHERE external_id IS NOT NULL / total` 의 일별 비율 (소수 1자리).

다른 핸들러는 변경 없음. 백엔드 응답 스키마/모델 변경 없음.

## 10. 의존성

추가 의존성 0. CDN 변경:
- 기존 `cdn.tailwindcss.com` 유지.
- 신규 `fonts.googleapis.com` (Inter, JetBrains Mono).

`htmx` CDN 은 base.html 에서 제거할 수 있지만 (현재 사용처 없음) 향후 동적 행동을 고려해 유지.

## 11. 범위 밖 (YAGNI)

- 다크모드 토글
- 외부 차트 라이브러리 (Chart.js 등)
- 모바일 햄버거/드로어 — flex-wrap 으로 충분
- 검색/정렬/HTMX 인터랙션 추가
- DB 마이그레이션
- 백엔드 도메인 로직 변경

## 12. 검증

수동 검증으로 충분 (이 변경은 시각 변경이며 자동 회귀 부담을 정당화하지 않는다):
- `docker compose up -d` 후
- `GET /viewer/` → 200, 메트릭 카드 3개와 sparkline 렌더 확인
- `GET /viewer/papers` → 200, 필터 작동, 페이지네이션 작동
- `GET /viewer/papers/{id}` → 200, 메타 및 청크 표
- `GET /viewer/chunks/{id}` → 200, 코드 블록 가독성
- `GET /viewer/jobs` → 200, 실패 행 강조

브라우저 콘솔 에러 0, 1280×800 / 1024×768 / 768×600 세 해상도에서 깨짐 없음.
