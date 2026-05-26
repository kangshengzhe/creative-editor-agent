"""Exception class hierarchy for the Creative Editor Agent.

These exceptions are raised internally by the API gateway, the tools, and the
orchestrator. They are converted to a public :class:`ErrorResponse` payload at
the API boundary by :meth:`ErrorResponse.from_exception`.

Error code mapping (see ``design.md`` Error Handling section):

================================  ==============================
Exception                          Public ``status`` field
================================  ==============================
:class:`ValidationError`           ``VALIDATION_ERROR``
:class:`GenerationFailureError`    ``GENERATION_FAILURE``
:class:`DegradedFailureError`      ``DEGRADED_FAILURE``
:class:`CascadeFailureError`       ``CASCADE_FAILURE``
:class:`ToolFailureError`          internal only — never surfaced
================================  ==============================
"""

from __future__ import annotations

from typing import Any, Optional


class AgentError(Exception):
    """Base exception for all Creative Editor Agent errors.

    Attributes:
        message: Human-readable error message.
        request_id: The request identifier associated with the error.
            ``None`` when the error occurs before a request id is assigned
            (e.g. malformed JSON during initial parse).
        details: Additional structured fields rendered into the public
            error response under ``error.details``.
    """

    #: Public error code rendered into ``error.code``. Subclasses override.
    code: str = "AGENT_ERROR"

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.details: dict[str, Any] = dict(details) if details else {}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"{type(self).__name__}(code={self.code!r}, "
            f"message={self.message!r}, request_id={self.request_id!r}, "
            f"details={self.details!r})"
        )


# ---------------------------------------------------------------------------
# Validation errors (Requirements 1.2, 1.3, 1.4, 1.5, 1.8, 1.9)
# ---------------------------------------------------------------------------


class ValidationError(AgentError):
    """Creative_Brief validation failure.

    Sub-error codes (carried in :attr:`code`):

    - ``MISSING_FIELD``: required field is missing, ``None``, an empty string,
      or pure whitespace.
    - ``INVALID_ENUM``: enum-typed field has a value outside the allowed set.
    - ``INVALID_LENGTH``: string field length is outside the permitted range
      (e.g. ``campaign_topic`` must be 1-200 characters).
    - ``MALFORMED_JSON``: input could not be parsed as JSON or has a type
      mismatch with the schema.

    Attributes:
        code: One of the sub-error code constants above.
        field: Name of the offending field, or ``None`` for ``MALFORMED_JSON``.
        allowed_values: For ``INVALID_ENUM`` errors, the list of valid values.
    """

    # Sub-error code constants, also used as the public ``error.code``.
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_ENUM = "INVALID_ENUM"
    INVALID_LENGTH = "INVALID_LENGTH"
    MALFORMED_JSON = "MALFORMED_JSON"

    _ALLOWED_CODES = frozenset(
        {MISSING_FIELD, INVALID_ENUM, INVALID_LENGTH, MALFORMED_JSON}
    )

    def __init__(
        self,
        code: str,
        message: str,
        field: Optional[str] = None,
        allowed_values: Optional[list[Any]] = None,
        request_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        if code not in self._ALLOWED_CODES:
            raise ValueError(
                f"ValidationError code must be one of {sorted(self._ALLOWED_CODES)}, "
                f"got {code!r}"
            )
        merged: dict[str, Any] = dict(details) if details else {}
        if field is not None:
            merged.setdefault("field", field)
        if allowed_values is not None:
            merged.setdefault("allowed_values", list(allowed_values))
        super().__init__(message, request_id=request_id, details=merged)
        self.code = code
        self.field = field
        self.allowed_values: Optional[list[Any]] = (
            list(allowed_values) if allowed_values is not None else None
        )


# ---------------------------------------------------------------------------
# Top-level failure errors surfaced to the API boundary
# ---------------------------------------------------------------------------


class GenerationFailureError(AgentError):
    """Creative_Generator failed after the 2 permitted retries.

    Maps to Requirements 2.7 and 9.1. No partial result is returned.
    """

    code: str = "GENERATION_FAILURE"

    def __init__(
        self,
        message: str = "Creative_Generator failed after 2 retries",
        request_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, request_id=request_id, details=details)


class DegradedFailureError(AgentError):
    """Refill attempts exhausted with fewer than 3 candidates remaining.

    After 2 refill loops the post-filter candidate count is still ``< 3``.
    Maps to Requirement 7.7. The response carries a ``partial_result`` with
    the current candidate count and refill attempts.

    Attributes:
        candidates_after_filter: Number of candidates remaining after BLOCK
            filtering, expected to be ``0``, ``1``, or ``2``.
        refill_attempts: Number of refill rounds executed (``0``, ``1``,
            or ``2``).
    """

    code: str = "DEGRADED_FAILURE"

    def __init__(
        self,
        candidates_after_filter: int,
        refill_attempts: int,
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        if message is None:
            message = (
                f"Only {candidates_after_filter} candidate(s) remain after "
                f"{refill_attempts} refill attempt(s); minimum 3 required."
            )
        merged: dict[str, Any] = dict(details) if details else {}
        merged.setdefault("candidates_after_filter", candidates_after_filter)
        merged.setdefault("refill_attempts", refill_attempts)
        super().__init__(message, request_id=request_id, details=merged)
        self.candidates_after_filter = candidates_after_filter
        self.refill_attempts = refill_attempts


class CascadeFailureError(AgentError):
    """Cumulative tool failures exceeded the global circuit-breaker threshold.

    Triggered when more than 5 tool invocations have failed within a single
    request (Requirement 9.6). The request is terminated immediately.

    Attributes:
        failure_count: Total tool failures observed when the breaker tripped
            (``> 5``).
    """

    code: str = "CASCADE_FAILURE"

    def __init__(
        self,
        failure_count: int,
        message: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        if message is None:
            message = (
                f"Tool failure count {failure_count} exceeded the threshold of 5; "
                "request terminated."
            )
        merged: dict[str, Any] = dict(details) if details else {}
        merged.setdefault("failure_count", failure_count)
        super().__init__(message, request_id=request_id, details=merged)
        self.failure_count = failure_count


# ---------------------------------------------------------------------------
# Internal-only tool failure (never surfaces to the user)
# ---------------------------------------------------------------------------


class ToolFailureError(AgentError):
    """Single tool invocation failed.

    Raised by individual tools (Compliance_Checker, Localization_Tool, etc.)
    when an LLM call, timeout, or other transient error occurs. The
    orchestrator catches this exception and applies the per-tool degradation
    policy described in Requirements 9.2 – 9.5, while accumulating a counter
    for the global circuit breaker (Requirement 9.6).

    This error must not reach the API response — :meth:`ErrorResponse.from_exception`
    will refuse to map it.

    Attributes:
        tool_name: Identifier of the failing tool (e.g. ``"Compliance_Checker"``).
        original_exception: The underlying exception that triggered the failure,
            if available.
    """

    code: str = "TOOL_FAILURE"

    def __init__(
        self,
        tool_name: str,
        message: str,
        original_exception: Optional[BaseException] = None,
        request_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        merged: dict[str, Any] = dict(details) if details else {}
        merged.setdefault("tool_name", tool_name)
        if original_exception is not None:
            merged.setdefault(
                "original_exception",
                f"{type(original_exception).__name__}: {original_exception}",
            )
        super().__init__(message, request_id=request_id, details=merged)
        self.tool_name = tool_name
        self.original_exception = original_exception
