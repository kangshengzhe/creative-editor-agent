"""Top-level Orchestrator: tool sequencing + refill loop + circuit breaker.

Implements design.md § Architecture / Orchestrator and Requirements 2.7,
7.5, 7.6, 7.7, 9.1, 9.6.

Flow
----

1. Load Platform_Spec from ``brief.target_platform``.
2. Initialise shared state (``tool_failure_counter`` for the global breaker,
   ``all_processed`` for accumulated candidates, ``total_generated``,
   ``refill_count``).
3. Generation + pipeline + refill loop (up to 1 initial + 2 refill rounds):
   a. Call ``Creative_Generator.generate`` (it does its own 2 retries
      internally, Req 2.7). On the *first* round any failure is fatal —
      :class:`GenerationFailureError` propagates with the request id. On
      *refill* rounds a generation failure is tolerated when the agent
      already has ≥ 3 compliant candidates: we stop refilling and proceed
      to ranking.
   b. Run the per-candidate pipeline in parallel via :func:`asyncio.gather`.
   c. Trip the global breaker if ``tool_failure_counter[0] > 5``
      (Requirement 9.6).
   d. Stop the loop as soon as we have ≥ 3 candidates whose final
      compliance report has no BLOCK violation.
4. If we still have < 3 compliant candidates after 3 rounds, raise
   :class:`DegradedFailureError` (Req 7.7) with the partial-result counts.
5. Call :func:`rank_candidates` to filter, score, sort, and package the
   final :class:`AB_Ranking`.

Notes
-----

* The pipeline never raises — every per-tool failure is captured into
  ``candidate.warnings`` and the shared breaker counter. The Orchestrator
  therefore relies on the counter (not exception flow) to decide when to
  abort, except for the explicit ``GenerationFailureError`` from the
  generator and the ``CascadeFailureError`` we raise here.
* ``Platform_Spec`` lookup is injected for testability: the default uses
  the bundled :func:`load_platform_spec`, but tests can pass their own
  loader to substitute fixture data.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from creative_agent.config import load_platform_spec
from creative_agent.errors import (
    CascadeFailureError,
    DegradedFailureError,
    GenerationFailureError,
)
from creative_agent.models import (
    AB_Ranking,
    Compliance_Severity,
    Creative_Brief,
    Creative_Candidate,
)
from creative_agent.observability.logging import get_logger
from creative_agent.observability.trace import TraceRecorder, ToolCallTrace
from creative_agent.orchestrator.composite_scorer import rank_candidates
from creative_agent.orchestrator.pipeline import PipelineDeps, process_candidate

if TYPE_CHECKING:
    from creative_agent.models.platform_spec import Platform_Spec
    from creative_agent.tools.compliance_checker import ComplianceChecker
    from creative_agent.tools.creative_generator import CreativeGenerator
    from creative_agent.tools.cta_optimizer import CTAOptimizer
    from creative_agent.tools.keyword_embedder import KeywordEmbedder
    from creative_agent.tools.localization_tool import LocalizationTool

__all__ = ["Orchestrator"]

log = get_logger(__name__)


# Global circuit-breaker threshold (Requirement 9.6). The breaker trips when
# the cumulative tool-failure counter strictly exceeds this value.
_CASCADE_FAILURE_THRESHOLD: int = 30

# Refill policy (Requirement 7.6). One initial generation round plus up to
# two refill rounds = three rounds in total.
_MAX_ROUNDS: int = 3

# Minimum compliant candidates we must surface in the AB_Ranking
# (Requirement 7.6 / 7.7).
_MIN_COMPLIANT_CANDIDATES: int = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Orchestrator:
    """Top-level orchestration of generation + pipeline + ranking.

    The orchestrator is stateless across requests; per-request state lives
    on the stack inside :meth:`orchestrate`.

    Args:
        creative_generator: Generates candidate copies from a brief.
        compliance_checker: Used by the per-candidate pipeline.
        localization_tool: Used by the per-candidate pipeline.
        keyword_embedder: Used by the per-candidate pipeline.
        cta_optimizer: Used by the per-candidate pipeline.
        platform_loader: Optional override for ``load_platform_spec``. The
            production path uses the bundled JSON configs; tests inject a
            fixture lookup.
    """

    def __init__(
        self,
        creative_generator: "CreativeGenerator",
        compliance_checker: "ComplianceChecker",
        localization_tool: "LocalizationTool",
        keyword_embedder: "KeywordEmbedder",
        cta_optimizer: "CTAOptimizer",
        platform_loader: Optional[Callable[[Any], "Platform_Spec"]] = None,
        trace_recorder: Optional[TraceRecorder] = None,
    ) -> None:
        self._creative_generator = creative_generator
        self._compliance_checker = compliance_checker
        self._localization_tool = localization_tool
        self._keyword_embedder = keyword_embedder
        self._cta_optimizer = cta_optimizer
        self._platform_loader = platform_loader or load_platform_spec
        self._trace = trace_recorder or TraceRecorder()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def orchestrate(
        self,
        brief: Creative_Brief,
        request_id: str,
        warnings: Optional[list[str]] = None,
    ) -> AB_Ranking:
        """Run the full pipeline and return the ranked candidates.

        Args:
            brief: Validated :class:`Creative_Brief` from the API gateway.
            request_id: Per-request identifier.
            warnings: Request-level warnings forwarded to the response
                (e.g. ``"keywords truncated from 30 to 20"``). Optional.

        Returns:
            :class:`AB_Ranking` with at least 3 compliant candidates.

        Raises:
            GenerationFailureError: When Creative_Generator cannot deliver
                an initial batch of candidates after its own 2 retries
                (Req 2.7 / 9.1).
            CascadeFailureError: When cumulative tool failures exceed the
                global threshold of 5 (Req 9.6).
            DegradedFailureError: When 3 rounds of generation + pipeline
                still leave fewer than 3 compliant candidates (Req 7.7).
        """
        platform_spec = self._platform_loader(brief.target_platform)

        # Start tracing this request
        self._trace.start_request(request_id, brief=brief)

        tool_failure_counter: list[int] = [0]
        pipeline_deps = PipelineDeps(
            compliance_checker=self._compliance_checker,
            localization_tool=self._localization_tool,
            keyword_embedder=self._keyword_embedder,
            cta_optimizer=self._cta_optimizer,
            request_id=request_id,
            tool_failure_counter=tool_failure_counter,
        )

        all_processed: list[Creative_Candidate] = []
        total_generated = 0
        refill_count = 0
        start_time = time.perf_counter()

        log.info(
            "orchestrator.start",
            request_id=request_id,
            target_platform=brief.target_platform.value,
            target_market=brief.target_market.value,
            creative_type=brief.creative_type.value,
            keyword_count=len(brief.keywords or []),
        )

        for refill_round in range(_MAX_ROUNDS):
            # ------------------------------------------------------------------
            # Stage A: generate candidates
            # ------------------------------------------------------------------
            exclude = [c.source_copy for c in all_processed]
            try:
                gen_output = await self._creative_generator.generate(
                    brief=brief,
                    platform_spec=platform_spec,
                    exclude_copies=exclude,
                    min_count=5,
                    request_id=request_id,
                    tool_failure_counter=tool_failure_counter,
                )
            except GenerationFailureError as exc:
                # First-round generation failure is fatal (Req 9.1).
                if refill_round == 0:
                    log.error(
                        "orchestrator.generation_failure",
                        request_id=request_id,
                        refill_round=refill_round,
                        error=exc.message,
                    )
                    # Re-raise with our request id stamped in for ErrorResponse.
                    if exc.request_id is None:
                        exc.request_id = request_id
                    raise

                # Refill-round generation failure is tolerated *iff* we
                # already have enough compliant candidates to satisfy 7.6.
                compliant = _count_compliant(all_processed)
                if compliant >= _MIN_COMPLIANT_CANDIDATES:
                    log.warning(
                        "orchestrator.refill_generation_failure_tolerated",
                        request_id=request_id,
                        refill_round=refill_round,
                        compliant_count=compliant,
                        error=exc.message,
                    )
                    break

                log.error(
                    "orchestrator.refill_generation_failure_fatal",
                    request_id=request_id,
                    refill_round=refill_round,
                    compliant_count=compliant,
                    error=exc.message,
                )
                if exc.request_id is None:
                    exc.request_id = request_id
                raise

            total_generated += len(gen_output.candidates)

            # Trip the breaker as early as possible (Req 9.6).
            self._check_cascade(tool_failure_counter[0], request_id)

            # ------------------------------------------------------------------
            # Stage B: per-candidate pipeline (parallel fan-out)
            # ------------------------------------------------------------------
            if gen_output.candidates:
                processed = await asyncio.gather(
                    *[
                        process_candidate(c, brief, platform_spec, pipeline_deps)
                        for c in gen_output.candidates
                    ]
                )
                all_processed.extend(processed)

            # Re-check the breaker once the pipeline has had a chance to
            # log its own failures.
            self._check_cascade(tool_failure_counter[0], request_id)

            compliant_count = _count_compliant(all_processed)

            log.info(
                "orchestrator.generation_round",
                request_id=request_id,
                refill_round=refill_round,
                generated_in_round=len(gen_output.candidates),
                total_generated=total_generated,
                compliant_count=compliant_count,
                tool_failures=tool_failure_counter[0],
            )

            # We have enough compliant candidates — exit the loop.
            if compliant_count >= _MIN_COMPLIANT_CANDIDATES:
                break

            # Otherwise count this as a refill attempt (rounds 1..N-1 are
            # refills; round 0 is the initial attempt). The post-loop check
            # below decides whether to raise DegradedFailureError.
            refill_count = refill_round + 1

        # ----------------------------------------------------------------------
        # Stage C: degraded-failure check (Req 7.7)
        # ----------------------------------------------------------------------
        compliant_count = _count_compliant(all_processed)
        if compliant_count < _MIN_COMPLIANT_CANDIDATES:
            log.error(
                "orchestrator.degraded_failure",
                request_id=request_id,
                compliant_count=compliant_count,
                refill_attempts=refill_count,
                total_generated=total_generated,
            )
            raise DegradedFailureError(
                candidates_after_filter=compliant_count,
                refill_attempts=refill_count,
                request_id=request_id,
            )

        # ----------------------------------------------------------------------
        # Stage D: rank
        # ----------------------------------------------------------------------
        generation_time_ms = int((time.perf_counter() - start_time) * 1000)
        ranking = rank_candidates(
            candidates=all_processed,
            request_id=request_id,
            refill_count=refill_count,
            generation_time_ms=generation_time_ms,
            warnings=warnings or [],
            brief_summary={
                "topic": brief.campaign_topic,
                "platform": brief.target_platform.value,
                "market": brief.target_market.value,
                "type": brief.creative_type.value,
            },
            total_generated=total_generated,
        )

        log.info(
            "orchestrator.completed",
            request_id=request_id,
            ranked_count=len(ranking.ranked_candidates),
            total_generated=total_generated,
            refill_count=refill_count,
            generation_time_ms=generation_time_ms,
            tool_failures=tool_failure_counter[0],
        )

        # Finalize trace — persist to disk
        await self._trace.finalize(
            request_id=request_id,
            generation_time_ms=generation_time_ms,
            result_status="OK",
        )

        return ranking

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_cascade(failure_count: int, request_id: str) -> None:
        """Trip the global circuit breaker when failures exceed the threshold."""
        if failure_count > _CASCADE_FAILURE_THRESHOLD:
            log.error(
                "orchestrator.cascade_failure",
                request_id=request_id,
                failure_count=failure_count,
                threshold=_CASCADE_FAILURE_THRESHOLD,
            )
            raise CascadeFailureError(
                failure_count=failure_count,
                request_id=request_id,
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _has_block(candidate: Creative_Candidate) -> bool:
    """Return True iff the candidate's compliance report carries a BLOCK."""
    return any(
        v.severity == Compliance_Severity.BLOCK
        for v in candidate.compliance_report.violations
    )


def _count_compliant(candidates: list[Creative_Candidate]) -> int:
    """Count candidates whose final compliance report has no BLOCK."""
    return sum(1 for c in candidates if not _has_block(c))
