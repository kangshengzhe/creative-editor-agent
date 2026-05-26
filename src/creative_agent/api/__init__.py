"""API gateway: request validation, ID assignment, top-level handler.

Public surface:

- :func:`parse_and_validate` — parse a raw request payload, allocate a
  request id, persist the original brief, and return a validated
  :class:`Creative_Brief` plus any non-fatal warnings.
- :func:`reset_request_counter` — test-only helper to reset the
  request-id sequence counter.
- :func:`handle_request` — top-level entry that wires the gateway to the
  orchestrator and converts errors to ``ErrorResponse`` payloads.
- :func:`build_handler` — bind ``handle_request`` to a specific
  :class:`Orchestrator` instance (HTTP framework-friendly).
"""

from creative_agent.api.gateway import (
    parse_and_validate,
    reset_request_counter,
)
from creative_agent.api.handler import build_handler, handle_request

__all__ = [
    "build_handler",
    "handle_request",
    "parse_and_validate",
    "reset_request_counter",
]
