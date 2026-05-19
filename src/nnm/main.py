from __future__ import annotations
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app

from nnm.api.viewer.router import router as viewer_router
from nnm.config import get_settings
from nnm.errors import register_exception_handlers
from nnm.infra.local_embedder import LocalEmbedder
from nnm.logging import configure_logging

log = structlog.get_logger()


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

    register_exception_handlers(app)
    app.include_router(viewer_router, prefix="/viewer", tags=["viewer"])
    app.mount("/metrics", make_asgi_app())

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
