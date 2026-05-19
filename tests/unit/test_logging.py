from __future__ import annotations
import logging
import structlog
from nnm.logging import configure_logging


def test_configure_logging(monkeypatch):
    monkeypatch.setenv("NNM_DB_URL", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setenv("NNM_S3_BUCKET", "b")
    monkeypatch.setenv("NNM_LOG_LEVEL", "DEBUG")
    from nnm.config import get_settings
    get_settings.cache_clear()

    configure_logging()
    log = structlog.get_logger()
    log.info("test_event", key="value")
    assert logging.getLogger().level == logging.DEBUG
