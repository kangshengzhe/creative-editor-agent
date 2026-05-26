"""Pydantic models for the unified error response payload.

The API gateway converts any :class:`AgentError` into an :class:`ErrorResponse`
via :meth:`ErrorResponse.from_exception`. The shape mirrors the
``ErrorResponse`` interface defined in ``design.md``::

    {
      "request_id": "req_<timestamp>_<seq>",
      "status": "VALIDATION_ERROR" | "GENERATION_FAILURE"
              | "DEGRADED_FAILURE" | "CASCADE_FAILURE",
      "error": {
        "code": "MISSING_FIELD" | ... ,
        "message": "...",
        "details": { ... }
      },
      "partial_result": {              // only for DEGRADED_FAILURE
        "candidates_after_filter": 2,
        "refill_attempts": 2
      }
    }
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from creative_agent.errors.codes import (
    AgentError,
    CascadeFailureError,
    DegradedFailureError,
    GenerationFailureError,
    ToolFailureError,
    ValidationError,
)

#: Public ``status`` enum exposed in the error response.
ErrorStatus = Literal[
    "VALIDATION_ERROR",
    "GENERATION_FAILURE",
    "DEGRADED_FAILURE",
    "CASCADE_FAILURE",
]


class ErrorDetail(BaseModel):
    """Structured error body attached to every :class:`ErrorResponse`."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description=(
            "Public error code. For VALIDATION_ERROR this is one of "
            "MISSING_FIELD, INVALID_ENUM, INVALID_LENGTH, MALFORMED_JSON; "
            "for the other statuses it equals the status string."
        ),
    )
    message: str = Field(..., description="Human-readable error message.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional structured fields, e.g. ``field``, ``allowed_values``, "
            "``tool_name``, ``failure_count``."
        ),
    )


class PartialResult(BaseModel):
    """Partial result payload returned alongside ``DEGRADED_FAILURE``.

    Mirrors the partial result described in Requirement 7.7 — the caller can
    see how many candidates survived BLOCK filtering and how many refill
    attempts were executed before giving up.
    """

    model_config = ConfigDict(extra="forbid")

    candidates_after_filter: int = Field(
        ...,
        ge=0,
        description="Number of candidates remaining after BLOCK filtering.",
    )
    refill_attempts: int = Field(
        ...,
        ge=0,
        description="Number of refill rounds executed before failure.",
    )


class ErrorResponse(BaseModel):
    """Unified error response payload returned by the API gateway."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        ...,
        description="Request identifier; empty string when failure occurred "
        "before id assignment (e.g. JSON parse error).",
    )
    status: ErrorStatus
    error: ErrorDetail
    partial_result: Optional[PartialResult] = None

    @classmethod
    def from_exception(cls, exc: AgentError) -> "ErrorResponse":
        """Build an :class:`ErrorResponse` from a raised :class:`AgentError`.

        Mapping rules:

        - :class:`ValidationError` -> ``VALIDATION_ERROR`` with ``error.code``
          set to the sub-error code (``MISSING_FIELD`` etc.).
        - :class:`GenerationFailureError` -> ``GENERATION_FAILURE``.
        - :class:`DegradedFailureError` -> ``DEGRADED_FAILURE`` with a
          populated ``partial_result``.
        - :class:`CascadeFailureError` -> ``CASCADE_FAILURE``.

        :class:`ToolFailureError` must be handled internally by the
        orchestrator; passing one to this method raises ``ValueError``
        because tool failures are not exposed in the public response.
        """
        if isinstance(exc, ToolFailureError):
            raise ValueError(
                "ToolFailureError is internal-only and must not be surfaced "
                "as an ErrorResponse; the orchestrator must degrade or "
                "convert it to a top-level failure."
            )

        status, code = _resolve_status_and_code(exc)

        partial_result: Optional[PartialResult] = None
        if isinstance(exc, DegradedFailureError):
            partial_result = PartialResult(
                candidates_after_filter=exc.candidates_after_filter,
                refill_attempts=exc.refill_attempts,
            )

        return cls(
            request_id=exc.request_id or "",
            status=status,
            error=ErrorDetail(
                code=code,
                message=exc.message,
                details=dict(exc.details),
            ),
            partial_result=partial_result,
        )


def _resolve_status_and_code(exc: AgentError) -> tuple[ErrorStatus, str]:
    """Map an :class:`AgentError` instance to ``(status, code)``."""
    if isinstance(exc, ValidationError):
        return "VALIDATION_ERROR", exc.code
    if isinstance(exc, GenerationFailureError):
        return "GENERATION_FAILURE", GenerationFailureError.code
    if isinstance(exc, DegradedFailureError):
        return "DEGRADED_FAILURE", DegradedFailureError.code
    if isinstance(exc, CascadeFailureError):
        return "CASCADE_FAILURE", CascadeFailureError.code
    raise ValueError(
        f"Cannot map exception {type(exc).__name__} to a public ErrorResponse; "
        "only ValidationError, GenerationFailureError, DegradedFailureError, "
        "and CascadeFailureError are supported."
    )
