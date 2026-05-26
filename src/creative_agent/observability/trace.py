"""Trace recording for Creative Editor Agent.

Implements design.md § Components / Observability Layer and Requirements
8.2 — 8.5.

Pieces in this module:

* :class:`ToolCallTrace` / :class:`RequestTrace` — Pydantic models
  persisted to ``traces/<request_id>/trace.json`` (Req 8.3). The brief is
  carried verbatim on :class:`RequestTrace` so it survives any failure
  path (Req 9.7).
* :class:`TraceRecorder` — collects per-tool-call traces during a request
  and finalises the request-level :class:`RequestTrace`, persisting it to
  ``traces/<request_id>/trace.json``. Methods are intentionally narrow:
  :meth:`start_request`, :meth:`add_tool_call`, :meth:`finalize`,
  :meth:`get_trace`.
* :func:`trace_tool` — async/sync tool decorator that emits a
  ``tool.invoked`` ``structlog`` log line carrying ``tool_name``,
  ``duration_ms``, and ``status``. The decorator deliberately does *not*
  reach into a recorder; the orchestrator owns that wiring and writes
  trace records explicitly with full input/output payloads.
* :func:`aggregate_by_dimension` — aggregate ``generation_time_ms``
  across an in-memory list of :class:`RequestTrace` along
  ``target_platform`` / ``target_market`` / ``creative_type`` (Req 8.5).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import statistics
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from creative_agent.models import Creative_Brief
from creative_agent.observability.logging import get_logger

import os

__all__ = [
    "ToolCallTrace",
    "RequestTrace",
    "TraceRecorder",
    "AggregateStats",
    "trace_tool",
    "aggregate_by_dimension",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default trace dir — overridden by ``CREATIVE_AGENT_TRACE_DIR`` env var,
# matching the behaviour the API gateway already uses for brief persistence.
_DEFAULT_TRACE_DIR: str = "./traces"


ToolCallStatus = Literal["OK", "ERROR", "TIMEOUT"]
"""Status of a single tool invocation."""


# ---------------------------------------------------------------------------
# Pydantic data models (mirrors design.md → Trace / Observability Record)
# ---------------------------------------------------------------------------


class ToolCallTrace(BaseModel):
    """Trace record of a single tool invocation."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        ..., description="Per-request id propagated from the API gateway."
    )
    tool_name: str = Field(
        ...,
        description="Tool identifier (e.g. ``Creative_Generator``, "
        "``Compliance_Checker``).",
    )
    input_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Representation of the tool's positional and keyword "
        "arguments, captured by the caller.",
    )
    output_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Representation of the tool's return value; empty "
        "dict on failure.",
    )
    status: str = Field(
        ..., description="``OK`` / ``ERROR`` / ``TIMEOUT``."
    )
    duration_ms: int = Field(
        ..., ge=0, description="Wall-clock duration of the tool invocation."
    )
    error: Optional[dict[str, Any]] = Field(
        default=None,
        description="``{type, message, stack}`` when ``status != OK``.",
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp at the moment the call completed.",
    )


class RequestTrace(BaseModel):
    """Trace record for an entire request — aggregates all tool calls."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="Per-request id (Req 1.7).")
    brief: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Original ``Creative_Brief`` payload, persisted verbatim per "
            "Requirement 9.7 so it never gets lost on failure paths. "
            "``None`` until :meth:`TraceRecorder.start_request` binds it."
        ),
    )
    tool_calls: list[ToolCallTrace] = Field(
        default_factory=list,
        description="Tool invocations recorded during this request.",
    )
    generation_time_ms: int = Field(
        default=0,
        ge=0,
        description="End-to-end request duration in milliseconds (Req 8.5).",
    )
    result_status: str = Field(
        default="",
        description=(
            "Final outcome label set at finalisation: ``OK``, "
            "``DEGRADED_FAILURE``, etc. Empty while the request is in flight."
        ),
    )


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Per-process trace accumulator and persister.

    A single :class:`TraceRecorder` instance is shared by the agent for the
    lifetime of the process; per-request state is keyed by ``request_id``
    inside an in-memory dict guarded by an :class:`asyncio.Lock`.

    Lifecycle:

    1. :meth:`start_request` — called by the API gateway after the brief
       has been parsed. Creates an empty :class:`RequestTrace` keyed by
       ``request_id``.
    2. :meth:`add_tool_call` — called by the orchestrator (or an
       instrumented tool) to append a :class:`ToolCallTrace`.
    3. :meth:`finalize` — called once the response is ready. Records the
       end-to-end duration, status, and persists the trace to
       ``<base_dir>/<request_id>/trace.json``.
    4. :meth:`get_trace` — read-only access for tests / aggregation.

    Args:
        base_dir: Directory under which ``<request_id>/trace.json`` files
            are written by :meth:`finalize`. Defaults to the
            ``CREATIVE_AGENT_TRACE_DIR`` env var, then ``./traces``.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir: Path = (
            Path(base_dir)
            if base_dir is not None
            else Path(
                os.environ.get(
                    "CREATIVE_AGENT_TRACE_DIR", _DEFAULT_TRACE_DIR
                )
            )
        )
        self._traces: dict[str, RequestTrace] = {}
        self._start_times: dict[str, float] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_request(
        self,
        request_id: str,
        brief: Creative_Brief | None = None,
    ) -> None:
        """Begin tracing ``request_id``.

        Initialises an empty :class:`RequestTrace` and remembers the
        ``perf_counter`` baseline used by :meth:`finalize` to compute
        ``generation_time_ms``. The brief is serialised into a plain dict
        and stored verbatim (Req 9.7).
        """
        brief_payload: Optional[dict[str, Any]] = None
        if brief is not None:
            try:
                brief_payload = brief.model_dump(mode="json")
            except Exception:  # noqa: BLE001 — defensive; never crash here
                brief_payload = None
        self._traces[request_id] = RequestTrace(
            request_id=request_id, brief=brief_payload
        )
        self._start_times[request_id] = time.perf_counter()

    def add_tool_call(self, trace: ToolCallTrace) -> None:
        """Append ``trace`` to the in-memory record for its ``request_id``.

        If :meth:`start_request` was never called for that id (an unusual
        path — typically a tool fired before the gateway bound the brief),
        a fresh :class:`RequestTrace` is created on the fly so the call
        isn't lost.
        """
        record = self._traces.get(trace.request_id)
        if record is None:
            record = RequestTrace(request_id=trace.request_id)
            self._traces[trace.request_id] = record
            self._start_times.setdefault(
                trace.request_id, time.perf_counter()
            )
        record.tool_calls.append(trace)

    async def finalize(
        self,
        request_id: str,
        generation_time_ms: int,
        result_status: str,
    ) -> Path:
        """Stamp the request as finished and persist its trace to disk.

        Args:
            request_id: Request to finalise.
            generation_time_ms: End-to-end duration to record on the
                trace. Use ``-1`` to ask the recorder to compute it from
                the :meth:`start_request` baseline.
            result_status: Outcome label (``OK``, ``DEGRADED_FAILURE``,
                ``GENERATION_FAILURE``, ...).

        Returns:
            The target ``trace.json`` path. The write itself is best-effort:
            filesystem errors are logged but not propagated, since losing a
            trace must never crash an otherwise-successful request.
        """
        async with self._lock:
            record = self._traces.get(request_id)
            if record is None:
                record = RequestTrace(request_id=request_id)
                self._traces[request_id] = record

            if generation_time_ms < 0:
                start = self._start_times.get(request_id)
                if start is not None:
                    generation_time_ms = int(
                        (time.perf_counter() - start) * 1000
                    )
                else:
                    generation_time_ms = 0

            record.generation_time_ms = max(0, int(generation_time_ms))
            record.result_status = result_status

            target_dir = self._base_dir / request_id
            target_path = target_dir / "trace.json"
            try:
                # Persistence is synchronous I/O; offloaded so the event
                # loop stays responsive when traces grow large.
                await asyncio.to_thread(
                    _write_trace_to_disk, target_dir, target_path, record
                )
            except OSError as exc:
                log.warning(
                    "trace.persist_failed",
                    request_id=request_id,
                    path=str(target_path),
                    error=f"{type(exc).__name__}: {exc}",
                )
            return target_path

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    def get_trace(self, request_id: str) -> RequestTrace | None:
        """Return the in-memory trace for ``request_id``, or ``None``."""
        return self._traces.get(request_id)


def _write_trace_to_disk(
    target_dir: Path, target_path: Path, record: RequestTrace
) -> None:
    """Helper for :meth:`TraceRecorder.finalize`'s ``asyncio.to_thread`` call."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        record.model_dump_json(indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# trace_tool decorator
# ---------------------------------------------------------------------------


F = TypeVar("F", bound=Callable[..., Any])


def trace_tool(tool_name: str) -> Callable[[F], F]:
    """Decorate a tool callable so its invocation emits a structured log.

    The decorator is intentionally lightweight: it measures wall-clock
    duration, classifies the outcome (``OK`` / ``ERROR`` / ``TIMEOUT``),
    and emits a single ``tool.invoked`` ``structlog`` log line carrying
    ``tool_name``, ``duration_ms``, and ``status``. Per-request
    ``ToolCallTrace`` records are *not* written here — the orchestrator
    owns that wiring and feeds them to :meth:`TraceRecorder.add_tool_call`
    explicitly with full input/output payloads.

    Works for both ``async`` and synchronous functions; in this project
    all tools are async, but the sync branch keeps the helper usable for
    test stubs.
    """

    def decorator(func: F) -> F:
        is_coroutine = inspect.iscoroutinefunction(func)

        if is_coroutine:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                status: str = "OK"
                error: Optional[BaseException] = None
                try:
                    return await func(*args, **kwargs)
                except asyncio.TimeoutError as exc:
                    status = "TIMEOUT"
                    error = exc
                    raise
                except Exception as exc:  # noqa: BLE001 — re-raised
                    status = "ERROR"
                    error = exc
                    raise
                finally:
                    duration_ms = int((time.perf_counter() - start) * 1000)
                    _emit_tool_log(
                        tool_name=tool_name,
                        duration_ms=duration_ms,
                        status=status,
                        error=error,
                    )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            status: str = "OK"
            error: Optional[BaseException] = None
            try:
                return func(*args, **kwargs)
            except asyncio.TimeoutError as exc:
                status = "TIMEOUT"
                error = exc
                raise
            except Exception as exc:  # noqa: BLE001 — re-raised
                status = "ERROR"
                error = exc
                raise
            finally:
                duration_ms = int((time.perf_counter() - start) * 1000)
                _emit_tool_log(
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                    status=status,
                    error=error,
                )

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _emit_tool_log(
    *,
    tool_name: str,
    duration_ms: int,
    status: str,
    error: Optional[BaseException],
) -> None:
    """Emit the ``tool.invoked`` structlog line for a wrapped invocation."""
    if error is None:
        log.info(
            "tool.invoked",
            tool_name=tool_name,
            duration_ms=duration_ms,
            status=status,
        )
        return
    log.warning(
        "tool.invoked",
        tool_name=tool_name,
        duration_ms=duration_ms,
        status=status,
        error_type=type(error).__name__,
        error_message=str(error),
    )


# ---------------------------------------------------------------------------
# Aggregation (Requirement 8.5)
# ---------------------------------------------------------------------------


@dataclass
class AggregateStats:
    """Summary statistics for a bucket of request durations.

    All fields are derived from a list of ``generation_time_ms`` values
    bucketed by one of the routing dimensions on :class:`RequestTrace`'s
    ``brief``.
    """

    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


_VALID_DIMENSIONS = frozenset(
    {"target_platform", "target_market", "creative_type"}
)


def aggregate_by_dimension(
    traces: list[RequestTrace],
    dimension: Literal[
        "target_platform", "target_market", "creative_type"
    ],
) -> dict[str, AggregateStats]:
    """Aggregate ``generation_time_ms`` across ``traces`` by ``dimension``.

    For each distinct value of ``brief[dimension]``, returns the count,
    mean, and p50/p95/p99 of ``generation_time_ms``. Traces missing the
    requested dimension (or missing ``brief`` entirely) are bucketed under
    ``"unknown"``.

    Percentiles use the *nearest-rank* method on the sorted values:
    ``index = ceil(p * n) - 1``, clamped to ``[0, n-1]``. This matches the
    informal expectation of "the value at the Pth percentile" without
    interpolating, which is the common convention for SLA dashboards.

    Args:
        traces: In-memory list of :class:`RequestTrace`. Typically
            obtained via :meth:`TraceRecorder.get_trace` over a window.
        dimension: One of ``target_platform`` / ``target_market`` /
            ``creative_type`` — matches the keys present in the brief
            payload stored on the trace.

    Returns:
        Mapping from bucket value to its :class:`AggregateStats`. Empty
        when ``traces`` is empty.
    """
    if dimension not in _VALID_DIMENSIONS:
        raise ValueError(
            f"aggregate_by_dimension: 'dimension' must be one of "
            f"{sorted(_VALID_DIMENSIONS)}, got {dimension!r}"
        )

    buckets: dict[str, list[int]] = {}
    for trace in traces:
        bucket_key = _extract_dimension(trace, dimension)
        buckets.setdefault(bucket_key, []).append(
            int(trace.generation_time_ms)
        )

    return {
        bucket: _summarise_durations(values)
        for bucket, values in buckets.items()
    }


def _extract_dimension(trace: RequestTrace, dimension: str) -> str:
    """Pull ``dimension`` out of a trace's brief payload as a string key."""
    brief = trace.brief
    if not isinstance(brief, dict):
        return "unknown"
    raw = brief.get(dimension)
    if raw is None:
        return "unknown"
    text = str(raw).strip()
    if not text:
        return "unknown"
    return text


def _summarise_durations(values: list[int]) -> AggregateStats:
    """Compute :class:`AggregateStats` for a list of millisecond durations."""
    if not values:
        return AggregateStats(
            count=0,
            avg_ms=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
        )
    sorted_values = sorted(values)
    count = len(sorted_values)
    avg_ms = statistics.mean(sorted_values)
    return AggregateStats(
        count=count,
        avg_ms=float(avg_ms),
        p50_ms=float(_percentile(sorted_values, 0.50)),
        p95_ms=float(_percentile(sorted_values, 0.95)),
        p99_ms=float(_percentile(sorted_values, 0.99)),
        min_ms=float(sorted_values[0]),
        max_ms=float(sorted_values[-1]),
    )


def _percentile(sorted_values: list[int], p: float) -> int:
    """Nearest-rank percentile on a pre-sorted list of integers."""
    n = len(sorted_values)
    if n == 0:
        return 0
    # ``ceil(p * n) - 1`` clamped into [0, n-1].
    idx = int(-(-p * n // 1)) - 1  # ceil without importing math
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    return sorted_values[idx]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _format_exception(exc: BaseException) -> dict[str, Any]:
    """Render an exception into the ``{type, message, stack}`` triple."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "stack": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
