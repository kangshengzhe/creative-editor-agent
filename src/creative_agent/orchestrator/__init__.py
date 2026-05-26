"""Orchestrator: tool sequencing, parallel pipelines, composite scoring.

Currently implemented:

* Composite Scorer — :func:`compute_composite_score`, :func:`rank_candidates`
  (task 13.1).
* Per-candidate pipeline — :class:`PipelineDeps`, :func:`process_candidate`
  (task 15.1).
"""

from creative_agent.orchestrator.composite_scorer import (
    compute_composite_score,
    rank_candidates,
)
from creative_agent.orchestrator.orchestrator import Orchestrator
from creative_agent.orchestrator.pipeline import (
    PipelineDeps,
    process_candidate,
)

__all__ = [
    "Orchestrator",
    "PipelineDeps",
    "compute_composite_score",
    "process_candidate",
    "rank_candidates",
]
