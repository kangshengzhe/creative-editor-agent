"""Top-level request handler: API Gateway → Orchestrator → response.

Implements design.md § Components / 1. API Gateway top-level entry and
Requirements 1, 7.7, 8.1, 9.1, 9.6, 9.7.

Flow
----

1. ``parse_and_validate(raw_input)`` — allocate ``request_id``, persist the
   raw brief immediately (Req 9.7), then run schema validation.
2. ``Orchestrator.orchestrate(brief, request_id, warnings)`` — generate +
   pipeline + rank.
3. Serialise the resulting :class:`AB_Ranking` to a plain dict suitable
   for JSON HTTP response with a ``status_code`` of 200.
4. Catch :class:`AgentError` subclasses and convert them to
   :class:`ErrorResponse` payloads:

   * :class:`ValidationError`        → 400 ``VALIDATION_ERROR``
   * :class:`GenerationFailureError` → 502 ``GENERATION_FAILURE``
   * :class:`DegradedFailureError`   → 503 ``DEGRADED_FAILURE`` with
     ``partial_result``
   * :class:`CascadeFailureError`    → 503 ``CASCADE_FAILURE``

5. Anything else is wrapped as a 500 ``CASCADE_FAILURE``-shaped response
   with the original exception type/message in ``details`` — this never
   leaks a stack trace to the caller.

The function never raises; all error paths funnel through ``ErrorResponse``.
"""

from __future__ import annotations

from typing import Any, Union

from creative_agent.api.gateway import parse_and_validate
from creative_agent.errors import (
    AgentError,
    CascadeFailureError,
    DegradedFailureError,
    ErrorResponse,
    GenerationFailureError,
    ToolFailureError,
    ValidationError,
)
from creative_agent.errors.codes import AgentError as _AgentError  # type: ignore[unused-import]
from creative_agent.errors.responses import ErrorDetail
from creative_agent.observability.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from creative_agent.orchestrator import Orchestrator

__all__ = ["handle_request", "build_handler"]

log = get_logger(__name__)


# Status-code mapping (HTTP). Kept as a small constant table so the
# response shape stays predictable.
_STATUS_CODES: dict[str, int] = {
    "VALIDATION_ERROR": 400,
    "GENERATION_FAILURE": 502,
    "DEGRADED_FAILURE": 503,
    "CASCADE_FAILURE": 503,
}


def _wrap_response(
    *,
    body: dict[str, Any],
    status_code: int,
) -> dict[str, Any]:
    """Normalise the dict returned to callers.

    Wraps the success/error body inside a top-level dict that carries the
    HTTP status code. The HTTP server adapter is responsible for unpacking
    this into the actual response object; keeping the function transport-
    agnostic makes it trivially testable from unit tests.
    """
    return {"status_code": status_code, "body": body}


async def handle_request(
    raw_input: Union[str, bytes, dict],
    orchestrator: Orchestrator,
) -> dict[str, Any]:
    """Process a single Creative_Brief request end-to-end.

    Args:
        raw_input: Raw HTTP body — ``str``, ``bytes``, or a pre-decoded
            ``dict``. Forwarded as-is to :func:`parse_and_validate`.
        orchestrator: A fully-wired :class:`Orchestrator`. Lifecycle is
            owned by the caller; one instance is typically reused per
            process.

    Returns:
        A dict of the form ``{"status_code": int, "body": dict}``. The
        ``body`` is either the serialised :class:`AB_Ranking` (success) or
        the serialised :class:`ErrorResponse` (failure).
    """
    request_id: str = ""
    try:
        request_id, brief, warnings = await parse_and_validate(raw_input)
        bind_request_context(request_id=request_id)

        log.info(
            "handler.request_validated",
            request_id=request_id,
            warning_count=len(warnings),
        )

        ranking = await orchestrator.orchestrate(
            brief=brief,
            request_id=request_id,
            warnings=warnings,
        )

        body = ranking.model_dump(mode="json")
        log.info(
            "handler.request_completed",
            request_id=request_id,
            ranked_count=len(ranking.ranked_candidates),
            generation_time_ms=ranking.generation_time_ms,
        )
        return _wrap_response(body=body, status_code=200)

    except ValidationError as exc:
        return _handle_agent_error(exc, request_id, "validation_error")

    except GenerationFailureError as exc:
        return _handle_agent_error(exc, request_id, "generation_failure")

    except DegradedFailureError as exc:
        return _handle_agent_error(exc, request_id, "degraded_failure")

    except CascadeFailureError as exc:
        return _handle_agent_error(exc, request_id, "cascade_failure")

    except ToolFailureError as exc:
        # ToolFailureError must never reach here — the pipeline / orchestrator
        # convert it into degradation flags or higher-level errors. If we
        # see one it's a programming bug; surface a generic CASCADE_FAILURE
        # to the caller without leaking internals.
        log.error(
            "handler.tool_failure_leaked",
            request_id=request_id,
            tool_name=exc.tool_name,
            error_message=exc.message,
        )
        wrapped = CascadeFailureError(
            failure_count=1,
            message=(
                "Internal tool failure leaked to handler "
                f"(tool={exc.tool_name})"
            ),
            request_id=request_id or None,
        )
        return _handle_agent_error(wrapped, request_id, "cascade_failure")

    except AgentError as exc:
        # Defensive: any unmapped AgentError subclass.
        log.error(
            "handler.unmapped_agent_error",
            request_id=request_id,
            error_type=type(exc).__name__,
            error_message=exc.message,
        )
        return _handle_agent_error(exc, request_id, "agent_error")

    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        log.error(
            "handler.unexpected_error",
            request_id=request_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        # Build a synthetic CASCADE_FAILURE-shaped ErrorResponse so the
        # caller sees a stable contract instead of a stack trace.
        body = ErrorResponse(
            request_id=request_id or "",
            status="CASCADE_FAILURE",
            error=ErrorDetail(
                code="CASCADE_FAILURE",
                message="Internal server error",
                details={
                    "exception_type": type(exc).__name__,
                },
            ),
            partial_result=None,
        ).model_dump(mode="json")
        return _wrap_response(body=body, status_code=500)

    finally:
        clear_request_context()


def _handle_agent_error(
    exc: AgentError,
    request_id: str,
    log_event: str,
) -> dict[str, Any]:
    """Convert a public :class:`AgentError` to an HTTP response dict."""
    if not exc.request_id and request_id:
        exc.request_id = request_id

    log.warning(
        f"handler.{log_event}",
        request_id=exc.request_id or request_id,
        error_code=exc.code,
        error_message=exc.message,
        details=exc.details,
    )

    response = ErrorResponse.from_exception(exc)
    body = response.model_dump(mode="json", exclude_none=True)
    status_code = _STATUS_CODES.get(response.status, 500)
    return _wrap_response(body=body, status_code=status_code)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_handler(orchestrator: Orchestrator):
    """Return a coroutine ``handler(raw_input)`` bound to ``orchestrator``.

    Convenience for HTTP servers / framework adapters that want a single
    callable to register as their request handler.
    """

    async def _handler(raw_input: Union[str, bytes, dict]) -> dict[str, Any]:
        return await handle_request(raw_input, orchestrator)

    return _handler
