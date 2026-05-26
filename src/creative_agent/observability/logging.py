"""Structured logging configuration for Creative Editor Agent.

This module wires up `structlog` so that every log record is emitted as a
single line of JSON suitable for log aggregation (Elastic / Loki / CloudWatch),
and so that contextual fields (e.g. ``request_id``) can be bound once per
request via ``contextvars`` and automatically propagated to every subsequent
log call inside that request — even across ``asyncio`` task boundaries.

Conventional fields used across the codebase (per design doc / requirements
8.2 & 8.4):

* ``request_id``  — request trace id, bound at API gateway entry
* ``tool_name``   — name of the tool being invoked (Creative_Generator, ...)
* ``duration_ms`` — elapsed time of a tool call
* ``status``      — ``"OK"`` / ``"ERROR"`` / ``"TIMEOUT"``

Usage::

    from creative_agent import configure_logging, get_logger
    from creative_agent.observability.logging import bind_request_context

    configure_logging()                # once, at process start
    log = get_logger(__name__)

    bind_request_context(request_id="req_123")
    log.info("tool.invoked", tool_name="Creative_Generator")
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
    unbind_contextvars,
)
from structlog.stdlib import BoundLogger

__all__ = [
    "configure_logging",
    "get_logger",
    "bind_request_context",
    "unbind_request_context",
    "clear_request_context",
]


_CONFIGURED: bool = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Initialize structlog + stdlib logging for JSON line output.

    Idempotent: safe to call multiple times — subsequent calls only adjust the
    log level. Should be called once at application startup (API gateway entry,
    test fixtures, CLI entry point).

    Processor chain (in order):

    1. ``merge_contextvars`` — pull ``contextvars``-bound fields (e.g.
       ``request_id``) into the event dict so they appear on every record.
    2. ``add_log_level`` — inject the textual ``level`` field.
    3. ``TimeStamper(fmt="iso")`` — ISO 8601 timestamps in UTC.
    4. ``StackInfoRenderer`` — render ``stack_info=True`` calls as a string.
    5. ``format_exc_info`` — render ``exc_info`` to a printable traceback.
    6. ``JSONRenderer`` — final newline-delimited JSON output.
    """
    global _CONFIGURED

    # Resolve string levels ("INFO", "debug", ...) to numeric values.
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO

    # Route stdlib logging records through a single stream handler on stdout
    # so anything logged via plain `logging.getLogger()` is also JSON-rendered
    # by structlog (when wrap_for_formatter is wired up). For the Agent's own
    # code paths we use structlog directly, but third-party libs (httpx, etc.)
    # use stdlib logging — keeping a basic handler avoids "No handlers" warnings.
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        # Minimal formatter; structlog produces the structured payload itself.
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
    else:
        for h in root_logger.handlers:
            h.setLevel(level)

    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a structlog ``BoundLogger`` for the given module / component.

    Lazily configures logging with default settings if ``configure_logging``
    has not been called yet, so test code and ad-hoc scripts can simply do
    ``log = get_logger(__name__)`` without ceremony.

    :param name: typically ``__name__`` of the calling module; passed through
        to structlog as the ``logger`` field for filtering / routing.
    """
    if not _CONFIGURED:
        configure_logging()
    if name is None:
        return structlog.get_logger()
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Context binding helpers
# ---------------------------------------------------------------------------
# These wrap ``structlog.contextvars`` so callers don't need to import structlog
# directly. Bound values live in a ``contextvars.ContextVar`` so they are:
#   * isolated per asyncio Task (no cross-request leakage)
#   * automatically propagated into child tasks created via asyncio.gather etc.

def bind_request_context(**kwargs: Any) -> None:
    """Bind one or more fields (e.g. ``request_id``) to the current context.

    All subsequent log calls in this request / task will include these fields
    automatically. Typical usage at the API gateway::

        bind_request_context(request_id=request_id)
    """
    bind_contextvars(**kwargs)


def unbind_request_context(*keys: str) -> None:
    """Remove specific fields from the current logging context."""
    if keys:
        unbind_contextvars(*keys)


def clear_request_context() -> None:
    """Clear all context-bound fields. Call at the end of each request."""
    clear_contextvars()
