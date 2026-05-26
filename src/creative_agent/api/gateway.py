"""API Gateway: parse and validate Creative_Brief inputs.

This module is the entry point invoked before the Orchestrator. Its
responsibilities, drawn from Requirement 1 and the Design's *API Gateway*
section, are:

1. Allocate a process-lifetime unique, monotonically increasing
   ``request_id`` (Requirement 1.7).
2. Persist the raw input to ``traces/<request_id>/brief.json`` *immediately*
   after id assignment, so that Requirement 9.7 — original brief is never
   lost on subsequent failures — holds for any downstream exception.
3. Parse the input as JSON (when given as ``str`` / ``bytes``).
4. Run schema validation: required fields, enum membership, length bounds,
   keyword-list truncation.
5. Construct a :class:`Creative_Brief` Pydantic model, surfacing any leftover
   model-level failures as :class:`ValidationError`.

The whole pipeline is designed to complete well under the 200ms SLA
(Requirements 1.1 - 1.5, 1.8, 1.9): all steps are pure Python with at most
one synchronous file write.
"""

from __future__ import annotations

import json
import os
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any, Iterator, Union

from pydantic import ValidationError as PydanticValidationError

from creative_agent.errors import ValidationError
from creative_agent.models import (
    Creative_Brief,
    Creative_Type,
    Target_Market,
    Target_Platform,
)
from creative_agent.observability.logging import get_logger

__all__ = [
    "parse_and_validate",
    "reset_request_counter",
]

log = get_logger(__name__)

#: Required top-level fields per Requirement 1.2.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "campaign_topic",
    "target_platform",
    "target_market",
    "creative_type",
)

#: Inclusive bounds on ``campaign_topic`` per Requirement 1.9.
_TOPIC_MIN_LEN: int = 1
_TOPIC_MAX_LEN: int = 200

#: Hard cap on the number of keywords retained, per Requirement 1.6.
_KEYWORDS_LIMIT: int = 20

#: Default trace directory; overridable via ``CREATIVE_AGENT_TRACE_DIR``.
_DEFAULT_TRACE_DIR: str = "./traces"


# ---------------------------------------------------------------------------
# Request id allocation (Requirement 1.7)
# ---------------------------------------------------------------------------
# We use a single ``itertools.count`` guarded by a ``threading.Lock`` so that
# the resulting ids are strictly monotonic across threads — ``next()`` on a
# bare ``count`` happens to be atomic in CPython today, but the lock makes
# the intent explicit and survives interpreter changes / non-CPython runtimes.
_seq_counter: Iterator[int] = count()
_seq_lock: threading.Lock = threading.Lock()


def _allocate_request_id() -> str:
    """Return a fresh request id of the form ``req_<unix_ms>_<seq>``.

    The sequence number is process-local and resets only when
    :func:`reset_request_counter` is called (test-only). The timestamp is
    Unix epoch milliseconds.
    """
    with _seq_lock:
        seq = next(_seq_counter)
    timestamp_ms = int(time.time() * 1000)
    return f"req_{timestamp_ms}_{seq}"


def reset_request_counter() -> None:
    """Reset the sequence counter back to zero. **Test-only.**

    Production code must never call this — it would let two concurrent
    requests share an id.
    """
    global _seq_counter
    with _seq_lock:
        _seq_counter = count()


# ---------------------------------------------------------------------------
# Raw-input persistence (Requirement 9.7)
# ---------------------------------------------------------------------------
def _resolve_trace_dir() -> Path:
    """Resolve the base trace directory.

    Honours the ``CREATIVE_AGENT_TRACE_DIR`` environment variable; falls back
    to ``./traces`` (resolved against the current working directory). The
    path is *not* created here — only the per-request subdirectory is.
    """
    return Path(os.environ.get("CREATIVE_AGENT_TRACE_DIR", _DEFAULT_TRACE_DIR))


def _persist_raw_input(request_id: str, raw_input: Union[str, bytes, dict]) -> None:
    """Write the raw input to ``traces/<request_id>/brief.json``.

    Persistence happens *before* any field validation so that even malformed
    inputs (e.g. broken JSON) leave a forensic record. The file format is:

    * ``dict`` -> pretty-printed JSON
    * ``str``  -> written verbatim (whatever the caller sent us)
    * ``bytes`` -> written verbatim as bytes
    * other -> best-effort ``str()`` repr (defensive; the validator below
      will then reject the input)

    Filesystem errors are logged but **not** raised: failure to write a
    trace file must not mask the actual validation outcome.
    """
    try:
        trace_dir = _resolve_trace_dir() / request_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        brief_path = trace_dir / "brief.json"

        if isinstance(raw_input, dict):
            brief_path.write_text(
                json.dumps(raw_input, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif isinstance(raw_input, bytes):
            brief_path.write_bytes(raw_input)
        elif isinstance(raw_input, str):
            brief_path.write_text(raw_input, encoding="utf-8")
        else:  # pragma: no cover - defensive
            brief_path.write_text(repr(raw_input), encoding="utf-8")
    except OSError as exc:
        # Persistence failure is operationally interesting but must not
        # crash the request — log and continue.
        log.warning(
            "request.brief.persist_failed",
            request_id=request_id,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _parse_json(
    raw_input: Union[str, bytes, dict], request_id: str
) -> dict[str, Any]:
    """Coerce the raw input into a ``dict`` or raise ``MALFORMED_JSON``."""
    if isinstance(raw_input, dict):
        return raw_input

    if isinstance(raw_input, (str, bytes)):
        try:
            decoded = json.loads(raw_input)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                code=ValidationError.MALFORMED_JSON,
                message=f"Invalid JSON: {exc.msg}",
                request_id=request_id,
                details={
                    "position": exc.pos,
                    "line": exc.lineno,
                    "column": exc.colno,
                },
            ) from exc

        if not isinstance(decoded, dict):
            raise ValidationError(
                code=ValidationError.MALFORMED_JSON,
                message=(
                    "Expected JSON object at the top level, "
                    f"got {type(decoded).__name__}"
                ),
                request_id=request_id,
            )
        return decoded

    raise ValidationError(
        code=ValidationError.MALFORMED_JSON,
        message=(
            "Unsupported input type; expected str, bytes, or dict, got "
            f"{type(raw_input).__name__}"
        ),
        request_id=request_id,
    )


def _check_required_fields(data: dict[str, Any], request_id: str) -> None:
    """Enforce Requirement 1.2: required fields are present and non-empty."""
    for field in _REQUIRED_FIELDS:
        if field not in data:
            raise ValidationError(
                code=ValidationError.MISSING_FIELD,
                message=f"Required field '{field}' is missing",
                field=field,
                request_id=request_id,
            )
        value = data[field]
        if value is None:
            raise ValidationError(
                code=ValidationError.MISSING_FIELD,
                message=f"Required field '{field}' is null",
                field=field,
                request_id=request_id,
            )
        if isinstance(value, str) and value.strip() == "":
            raise ValidationError(
                code=ValidationError.MISSING_FIELD,
                message=(
                    f"Required field '{field}' is empty or whitespace-only"
                ),
                field=field,
                request_id=request_id,
            )


def _check_enum_fields(data: dict[str, Any], request_id: str) -> None:
    """Enforce Requirements 1.3, 1.4, 1.5: enum-typed fields use valid values."""
    enum_fields: tuple[tuple[str, type], ...] = (
        ("target_platform", Target_Platform),
        ("target_market", Target_Market),
        ("creative_type", Creative_Type),
    )
    for field, enum_cls in enum_fields:
        value = data[field]
        allowed = [member.value for member in enum_cls]
        # Accept both the raw enum-value string and an actual enum instance
        # (defensive — Pydantic might re-emit enums during round-trips).
        candidate = value.value if hasattr(value, "value") else value
        if candidate not in allowed:
            raise ValidationError(
                code=ValidationError.INVALID_ENUM,
                message=(
                    f"Field '{field}' has invalid value {value!r}; "
                    f"expected one of {allowed}"
                ),
                field=field,
                allowed_values=allowed,
                request_id=request_id,
            )


def _check_topic_length(data: dict[str, Any], request_id: str) -> None:
    """Enforce Requirement 1.9: ``campaign_topic`` length in [1, 200]."""
    topic = data["campaign_topic"]
    if not isinstance(topic, str):
        # Type mismatch -> treat as malformed input.
        raise ValidationError(
            code=ValidationError.MALFORMED_JSON,
            message=(
                "Field 'campaign_topic' must be a string, got "
                f"{type(topic).__name__}"
            ),
            field="campaign_topic",
            request_id=request_id,
        )
    actual = len(topic)
    if actual < _TOPIC_MIN_LEN or actual > _TOPIC_MAX_LEN:
        raise ValidationError(
            code=ValidationError.INVALID_LENGTH,
            message=(
                f"Field 'campaign_topic' length {actual} is out of range "
                f"[{_TOPIC_MIN_LEN}, {_TOPIC_MAX_LEN}]"
            ),
            field="campaign_topic",
            request_id=request_id,
            details={
                "min": _TOPIC_MIN_LEN,
                "max": _TOPIC_MAX_LEN,
                "actual": actual,
            },
        )


def _truncate_keywords(
    data: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    """Enforce Requirement 1.6: cap ``keywords`` at 20 entries.

    Returns a (possibly new) ``data`` dict with the truncated list. Adds an
    operator-visible warning describing how many entries were dropped.
    Non-list ``keywords`` values are passed through untouched — Pydantic
    will reject them downstream.
    """
    keywords = data.get("keywords")
    if not isinstance(keywords, list):
        return data
    if len(keywords) <= _KEYWORDS_LIMIT:
        return data

    original_len = len(keywords)
    dropped = original_len - _KEYWORDS_LIMIT
    truncated = list(keywords[:_KEYWORDS_LIMIT])
    warnings.append(
        f"keywords truncated from {original_len} to {_KEYWORDS_LIMIT}, "
        f"dropped {dropped} keywords"
    )
    return {**data, "keywords": truncated}


def _build_brief(data: dict[str, Any], request_id: str) -> Creative_Brief:
    """Construct a :class:`Creative_Brief`, normalising Pydantic errors."""
    try:
        return Creative_Brief.model_validate(data)
    except PydanticValidationError as exc:
        # Surface the *first* error so the response stays focused; the full
        # error list is preserved in ``details`` for debugging.
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = first.get("loc", ())
        field = ".".join(str(part) for part in loc) if loc else None
        msg = first.get("msg", "Pydantic validation failed")

        # Map common Pydantic error types to our explicit codes so that
        # callers see a stable taxonomy. Anything else falls back to
        # MALFORMED_JSON, which is the right umbrella for shape mismatches.
        err_type = first.get("type", "")
        if err_type in {"missing", "value_error.missing"}:
            code = ValidationError.MISSING_FIELD
        elif err_type.startswith("enum") or err_type == "literal_error":
            code = ValidationError.INVALID_ENUM
        elif "length" in err_type or "string_too" in err_type:
            code = ValidationError.INVALID_LENGTH
        else:
            code = ValidationError.MALFORMED_JSON

        raise ValidationError(
            code=code,
            message=f"{msg} (field={field})" if field else msg,
            field=field,
            request_id=request_id,
            details={"pydantic_errors": errors},
        ) from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def parse_and_validate(
    raw_input: Union[str, bytes, dict],
) -> tuple[str, Creative_Brief, list[str]]:
    """Parse and validate a Creative_Brief input.

    Steps, in order:

    1. Allocate a fresh ``request_id`` (Requirement 1.7).
    2. Persist the raw input to ``traces/<request_id>/brief.json``
       (Requirement 9.7) — done *before* validation so the original payload
       survives any downstream failure.
    3. Coerce the input to a ``dict`` (JSON-decoding strings / bytes).
       Failure -> :class:`ValidationError` with code ``MALFORMED_JSON``
       (Requirement 1.8).
    4. Check required fields (Requirement 1.2).
    5. Check enum fields (Requirements 1.3, 1.4, 1.5).
    6. Check ``campaign_topic`` length (Requirement 1.9).
    7. Truncate ``keywords`` to 20 entries (Requirement 1.6), recording a
       user-visible warning.
    8. Build the :class:`Creative_Brief` Pydantic model.

    :param raw_input: HTTP body as ``str`` / ``bytes`` or a pre-decoded
        ``dict``.
    :returns: ``(request_id, brief, warnings)`` triple. ``warnings`` may be
        empty; it is *not* an error channel — only populated for non-fatal
        normalizations (currently only keyword truncation).
    :raises ValidationError: with the appropriate sub-code on any
        validation failure.
    """
    request_id = _allocate_request_id()
    log.info("request.received", request_id=request_id)

    # Persist first, validate second — guarantees Requirement 9.7 even if
    # downstream validation explodes.
    _persist_raw_input(request_id, raw_input)

    data = _parse_json(raw_input, request_id)

    _check_required_fields(data, request_id)
    _check_enum_fields(data, request_id)
    _check_topic_length(data, request_id)

    warnings: list[str] = []
    data = _truncate_keywords(data, warnings)

    brief = _build_brief(data, request_id)

    log.info(
        "request.validated",
        request_id=request_id,
        target_platform=brief.target_platform.value,
        target_market=brief.target_market.value,
        creative_type=brief.creative_type.value,
        warning_count=len(warnings),
    )
    return request_id, brief, warnings
