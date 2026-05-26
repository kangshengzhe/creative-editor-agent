"""Creative Editor Agent — Coco AI multi-tool LLM orchestrator.

Top-level package. Re-exports the most common observability helpers so
application code can simply do::

    from creative_agent import configure_logging, get_logger

    configure_logging()
    log = get_logger(__name__)
"""

from creative_agent.observability.logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "configure_logging",
    "get_logger",
]
