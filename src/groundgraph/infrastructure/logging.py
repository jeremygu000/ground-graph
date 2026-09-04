"""Structured logging helpers for the API boundary."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars


def configure_json_logging(log_level: str = "INFO") -> None:
    """Configure stdlib logging + structlog for JSON output."""

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def bind_request_context(values: Mapping[str, str]) -> None:
    """Bind request-scoped fields into structlog contextvars."""

    bind_contextvars(**values)


def clear_request_context() -> None:
    """Clear request-scoped structlog contextvars."""

    clear_contextvars()
