from __future__ import annotations
import logging
import sys

import structlog

from nnm.config import get_settings


def configure_logging() -> None:
    s = get_settings()
    is_prod = s.env != "local"

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if is_prod
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    level = getattr(logging, s.log_level.upper())
    logging.basicConfig(stream=sys.stdout, level=level, force=True)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
