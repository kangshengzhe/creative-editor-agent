"""Observability: structured logging, trace recording, metrics aggregation.

Currently implemented: structured logging (task 1.2), trace recording with
per-tool decorator and dimension aggregation (task 14.1).
"""

from creative_agent.observability.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    unbind_request_context,
)
from creative_agent.observability.trace import (
    AggregateStats,
    RequestTrace,
    ToolCallTrace,
    TraceRecorder,
    aggregate_by_dimension,
    trace_tool,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "bind_request_context",
    "unbind_request_context",
    "clear_request_context",
    # Trace
    "ToolCallTrace",
    "RequestTrace",
    "TraceRecorder",
    "AggregateStats",
    "trace_tool",
    "aggregate_by_dimension",
]
