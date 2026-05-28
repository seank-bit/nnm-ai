from __future__ import annotations
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from starlette.responses import Response

from nnm.api.viewer.router import router as viewer_router
from nnm.config import get_settings
from nnm.errors import register_exception_handlers
from nnm.infra.local_embedder import LocalEmbedder
from nnm.logging import configure_logging

log = structlog.get_logger()


# 인터넷 백그라운드 스캐너 (PHP/WordPress/Laravel exploit, env leak 등) path.
# 정상 라우트(/viewer, /eval, /metrics, /healthz)와 겹치지 않는 패턴만 등록.
_SCANNER_PATTERNS: tuple[str, ...] = (
    "/.env", "/.git", "/.ds_store", "/.aws", "/.svn",
    "wp-admin", "wp-includes", "wp-content", "wp-login", "wp-config", "wordpress",
    "xmlrpc.php", "phpunit", "phpmyadmin", "eval-stdin",
    "/vendor/", "/cgi-bin/", "/boaform/", "/__debug__/",
    "/think/app/", "/manage/account/login", "/whm",
    "/login.htm", "/login.html", "/admin/index.html",
    "/index.php?", "/api/index.php", "/containers/json", "cscoe",
)


def _is_scanner_path(path: str) -> bool:
    if not path or path == "/":
        return False
    low = path.lower()
    if low.endswith(".php"):
        return True
    return any(pat in low for pat in _SCANNER_PATTERNS)


def create_app(*, load_model: bool = True) -> FastAPI:
    configure_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        embedder = LocalEmbedder(settings=settings)
        if load_model:
            embedder.load()
        app.state.embedder = embedder
        log.info("nnm.startup", env=settings.env, device=settings.embedding_device)
        yield
        log.info("nnm.shutdown")

    app = FastAPI(title="nnm-ai", lifespan=lifespan)

    # 봇 차단은 다른 미들웨어/라우터보다 먼저 동작해야 하므로 마지막에 등록
    # (Starlette 미들웨어 스택은 LIFO: 마지막 등록이 바깥쪽).
    @app.middleware("http")
    async def _cid(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            structlog.contextvars.clear_contextvars()

    @app.middleware("http")
    async def _block_scanners(request: Request, call_next):
        if _is_scanner_path(request.url.path):
            # 403 Forbidden. 라우터/예외 핸들러까지 가지 않고 즉시 종료.
            return Response(status_code=403)
        return await call_next(request)

    register_exception_handlers(app)
    app.include_router(viewer_router, prefix="/viewer", tags=["viewer"])
    app.mount("/metrics", make_asgi_app())

    # RAGAS 평가 결과 (CSV/PNG/HTML) 정적 서빙. /eval/ 로 접근.
    eval_dir = Path(settings.storage_root) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    try:
        from nnm.eval.viz import render_index
        render_index(eval_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("eval.index_skip", error=str(e))
    app.mount("/eval", StaticFiles(directory=eval_dir, html=True), name="eval")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
