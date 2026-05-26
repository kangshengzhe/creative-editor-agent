"""Error codes and unified error response models.

Public surface:

- :class:`AgentError` and its subclasses (raised internally and at the API
  boundary).
- :class:`ErrorResponse` and its component models (serialized to the caller).
"""

from creative_agent.errors.codes import (
    AgentError,
    CascadeFailureError,
    DegradedFailureError,
    GenerationFailureError,
    ToolFailureError,
    ValidationError,
)
from creative_agent.errors.responses import (
    ErrorDetail,
    ErrorResponse,
    ErrorStatus,
    PartialResult,
)

__all__ = [
    "AgentError",
    "CascadeFailureError",
    "DegradedFailureError",
    "ErrorDetail",
    "ErrorResponse",
    "ErrorStatus",
    "GenerationFailureError",
    "PartialResult",
    "ToolFailureError",
    "ValidationError",
]
