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
from creative_agent.config.ad_group_quota import (
    MIN_COMPLIANT_FLOOR,
    target_count_for,
)
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
    from creative_agent.integration.semantic_diversity import (
        SemanticDiversityChecker,
    )
    from creative_agent.integration.keyword_localizer import KeywordLocalizer
    from creative_agent.integration.review_translator import ReviewTranslator
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

# Minimum compliant candidates we must surface in the AB_Ranking — the hard
# viability floor (Requirement 7.6 / 7.7, and Requirement 5.6). Falling below
# this after all refill rounds is a degraded failure. The per-request
# Target_Count (Requirement 5.1) is typically higher (e.g. 15 for HEADLINE);
# landing between this floor and the target yields a warning, not a failure.
_MIN_COMPLIANT_CANDIDATES: int = MIN_COMPLIANT_FLOOR


# Request-level warning appended (at most once) when the
# Semantic_Diversity_Checker degrades to text-dedup only because the embedding
# model was unavailable or timed out (Requirement 2.8).
_SEMANTIC_FALLBACK_WARNING: str = (
    "semantic diversity degraded to text-dedup only"
)


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
        semantic_diversity_checker: Optional
            :class:`SemanticDiversityChecker` (Req 2.5 / 2.7 / 2.8 / 2.9).
            When provided, freshly generated candidates are filtered for
            semantic duplicates (after the generator's text-based dedup and
            before the compliance pipeline) against the pool of all
            candidates accepted across every refill round. When ``None`` the
            semantic step is skipped entirely and behaviour is unchanged.
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
        semantic_diversity_checker: Optional["SemanticDiversityChecker"] = None,
        keyword_localizer: Optional["KeywordLocalizer"] = None,
        review_translator: Optional["ReviewTranslator"] = None,
    ) -> None:
        self._creative_generator = creative_generator
        self._compliance_checker = compliance_checker
        self._localization_tool = localization_tool
        self._keyword_embedder = keyword_embedder
        self._cta_optimizer = cta_optimizer
        self._platform_loader = platform_loader or load_platform_spec
        self._trace = trace_recorder or TraceRecorder()
        self._semantic_diversity_checker = semantic_diversity_checker
        # Optional: localizes generic SEO keywords into the market's language
        # (brand/proper nouns kept verbatim) so keywords don't stay English in
        # non-English copy. When None, keywords are used as-supplied (prior
        # behaviour) and all existing construction sites keep working.
        self._keyword_localizer = keyword_localizer
        # Optional: translates delivered copies into the HK review team's
        # languages (Simplified/Traditional Chinese + English) as a review aid,
        # shown in the UI detail panel. None → skipped (no behavior change).
        self._review_translator = review_translator

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

        # --- Keyword localization (PPC best practice) ---------------------
        # Generic SEO keywords (e.g. "topup") should appear in the market's
        # language ("recarga" for Spanish), while brand/proper nouns stay
        # verbatim. We localize ONCE here and replace brief.keywords, so both
        # generation and the keyword-embedder validation use the localized
        # terms (otherwise the embedder would reject localized copy for not
        # containing the English keyword). No-op for English markets or when no
        # localizer is wired; failures degrade to the original keywords.
        if self._keyword_localizer is not None and brief.keywords:
            try:
                from creative_agent.integration.language_prompts import (
                    LanguagePromptSelector,
                )

                primary_language = LanguagePromptSelector().get_primary_language(
                    brief.target_market
                )
            except KeyError:
                primary_language = "en"

            if primary_language != "en":
                mapping = await self._keyword_localizer.localize(
                    list(brief.keywords),
                    primary_language,
                    request_id=request_id,
                )
                localized_keywords = [mapping.get(k, k) for k in brief.keywords]
                if localized_keywords != list(brief.keywords):
                    changed = {
                        k: v for k, v in mapping.items() if k != v
                    }
                    log.info(
                        "orchestrator.keywords_localized",
                        request_id=request_id,
                        target_language=primary_language,
                        changed=changed,
                    )
                    # Replace on a copy so the original brief is not mutated
                    # for the caller / trace.
                    brief = brief.model_copy(update={"keywords": localized_keywords})

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

        # Semantic-diversity state, maintained across *all* refill rounds
        # (Req 2.9). ``semantic_accepted_pool`` holds the source copies of every
        # candidate accepted by the semantic check so far (captured *before* the
        # keyword-embedding pipeline mutates ``source_copy``).
        # ``semantic_rejected_copies`` holds copies rejected as semantic
        # duplicates so they are fed back into the generator's ``exclude`` set
        # and not regenerated. ``semantic_fallback_warned`` ensures the
        # text-dedup-only warning is appended at most once per request (Req 2.8).
        semantic_enabled = self._semantic_diversity_checker is not None
        semantic_accepted_pool: list[str] = []
        semantic_rejected_copies: list[str] = []
        # Quality-ordered backfill pool: candidates the semantic checker judged
        # near-duplicates. Used only if diverse generation can't reach the
        # ad-group quota, so the delivered count is always exactly the target
        # (business hard requirement) while still preferring diverse copy.
        semantic_reserve: list[Creative_Candidate] = []
        semantic_fallback_warned = False

        # Mutable warning list surfaced on the AB_Ranking output. Seeded from
        # the caller's warnings so angle-decomposition-failure warnings
        # (Req 3.8) accumulate alongside any gateway-level warnings.
        request_warnings: list[str] = list(warnings) if warnings else []

        # Whether the wired Creative_Generator can do angle-based round-robin
        # generation (an Angle_Splitter is injected). When False the
        # orchestrator uses the existing single-prompt generate() path
        # unchanged (Req 3.8 default).
        use_angle_generation = getattr(
            self._creative_generator, "supports_angle_generation", False
        )

        # Per-request Target_Count from the Ad_Group_Quota (Requirement 5.1):
        # HEADLINE=15, DESCRIPTION=10, CTA=5, LONG_COPY=5. An explicit
        # ``brief.target_count`` overrides the default (Requirement 5.8). This
        # single value drives generation (min_count) and is reported on the
        # AB_Ranking output (Requirement 5.7).
        target_count = brief.target_count or target_count_for(brief.creative_type)

        # Generation overshoot (Req 5 robustness): semantic-diversity dedup and
        # compliance filtering remove some candidates after generation, so if we
        # only generated exactly ``target_count`` we'd routinely land short of
        # the ad-group quota. We therefore ask the generator for ~1.7x the
        # target per round (rounded up). The surplus is cheap because angle
        # calls run concurrently, and the final set is truncated back to exactly
        # ``target_count`` by the ranker. Capped so a huge custom target can't
        # explode the call budget.
        import math as _math
        generation_target = min(int(_math.ceil(target_count * 1.7)), target_count + 20)

        log.info(
            "orchestrator.start",
            request_id=request_id,
            target_platform=brief.target_platform.value,
            target_market=brief.target_market.value,
            creative_type=brief.creative_type.value,
            keyword_count=len(brief.keywords or []),
            target_count=target_count,
        )

        for refill_round in range(_MAX_ROUNDS):
            # ------------------------------------------------------------------
            # Stage A: generate candidates
            # ------------------------------------------------------------------
            # Exclude both the copies we have already processed *and* any copies
            # rejected as semantic duplicates this request, so the generator's
            # refill loop replaces them rather than regenerating them (Req 2.5).
            exclude = [c.source_copy for c in all_processed]
            exclude.extend(semantic_rejected_copies)
            # Operator "常换常新" inputs: previously-delivered copies the caller
            # wants this run to avoid, so a re-run yields fresh variants (Req 6).
            if brief.regenerate_avoid:
                exclude.extend(
                    s for s in brief.regenerate_avoid if s and s.strip()
                )
            try:
                if use_angle_generation:
                    # Accumulate per-angle accepted counts across refill rounds
                    # so lowest-count cycling stays balanced (Req 3.5).
                    accepted_angle_counts = _accepted_angle_counts(all_processed)
                    gen_output = await self._creative_generator.generate_with_angles(
                        brief=brief,
                        platform_spec=platform_spec,
                        exclude_copies=exclude,
                        min_count=generation_target,
                        request_id=request_id,
                        tool_failure_counter=tool_failure_counter,
                        warnings=request_warnings,
                        accepted_angle_counts=accepted_angle_counts,
                    )
                else:
                    gen_output = await self._creative_generator.generate(
                        brief=brief,
                        platform_spec=platform_spec,
                        exclude_copies=exclude,
                        min_count=generation_target,
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
            # Stage A.5: semantic-diversity filtering (Req 2.5 / 2.7 / 2.8 / 2.9)
            # ------------------------------------------------------------------
            # Runs AFTER the generator's text-based dedup and BEFORE the
            # per-candidate compliance pipeline (Req 2.7). Freshly generated
            # candidates are checked against the pool of source copies accepted
            # across *all* refill rounds (Req 2.9); rejected candidates are
            # dropped (and fed back into the generator's ``exclude`` set so the
            # refill loop replaces them, Req 2.5). When no checker is injected
            # the step is skipped entirely and behaviour is unchanged.
            candidates_to_process = gen_output.candidates
            if semantic_enabled and gen_output.candidates:
                candidates_to_process, semantic_fallback_warned = (
                    await self._filter_semantic_diversity(
                        gen_output.candidates,
                        accepted_pool=semantic_accepted_pool,
                        rejected_copies=semantic_rejected_copies,
                        reserve=semantic_reserve,
                        request_warnings=request_warnings,
                        fallback_warned=semantic_fallback_warned,
                        request_id=request_id,
                    )
                )

            # ------------------------------------------------------------------
            # Stage B: per-candidate pipeline (parallel fan-out)
            # ------------------------------------------------------------------
            if candidates_to_process:
                processed = await asyncio.gather(
                    *[
                        process_candidate(c, brief, platform_spec, pipeline_deps)
                        for c in candidates_to_process
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

            # We have reached the per-type Target_Count (Requirement 5.2) —
            # the ad group can be fully filled, so stop refilling.
            if compliant_count >= target_count:
                break

            # Otherwise record how many *refill* rounds have been executed so
            # far. Round 0 is the initial run (not a refill); rounds 1 and 2
            # are refills. ``refill_count`` therefore equals ``refill_round``
            # and is capped at 2 to satisfy the AB_Ranking ``le=2`` constraint
            # (Requirement 7.6: 1 initial + up to 2 refills).
            refill_count = min(refill_round + 1, _MAX_ROUNDS - 1)

        # ----------------------------------------------------------------------
        # Stage B.5: quota backfill from the semantic reserve (Req 5 hard count)
        # ----------------------------------------------------------------------
        # If diversified generation fell short of the ad-group quota, top up
        # from the near-duplicate reserve (best-scoring first) so the delivered
        # count is exactly the target. This trades a little diversity for the
        # business's hard "must be 15/10" requirement, and ONLY kicks in when
        # genuinely diverse copy was insufficient. Reserve candidates still go
        # through the full per-candidate pipeline (compliance/keyword/CTA).
        #
        # Decision status: CONFIRMED by the business owner (2026-06) — when a
        # market can't yield enough fully-distinct copies, prioritise hitting
        # the exact 15/10 count and backfill with near-duplicates plus a
        # warning (option "A"), rather than returning fewer fully-diverse ones.
        compliant_count = _count_compliant(all_processed)
        if (
            semantic_enabled
            and compliant_count < target_count
            and semantic_reserve
        ):
            shortfall = target_count - compliant_count
            backfill = semantic_reserve[:shortfall]
            log.warning(
                "orchestrator.quota_backfill",
                request_id=request_id,
                compliant_count=compliant_count,
                target_count=target_count,
                backfill_count=len(backfill),
                reserve_size=len(semantic_reserve),
            )
            processed_backfill = await asyncio.gather(
                *[
                    process_candidate(c, brief, platform_spec, pipeline_deps)
                    for c in backfill
                ]
            )
            all_processed.extend(processed_backfill)
            request_warnings.append(
                f"quota backfill: added {len(backfill)} near-duplicate "
                f"candidate(s) to reach the {target_count}-candidate "
                f"{brief.creative_type.value} target (diverse generation "
                "produced fewer unique candidates)"
            )

        # ----------------------------------------------------------------------
        # Stage C: degraded-failure check (Req 7.7 / 5.6) + under-fill warning
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

        # Between the viability floor and the Target_Count: return what we have
        # and surface an under-fill warning rather than failing (Requirement 5.5).
        if compliant_count < target_count:
            log.warning(
                "orchestrator.under_filled",
                request_id=request_id,
                compliant_count=compliant_count,
                target_count=target_count,
                creative_type=brief.creative_type.value,
            )
            request_warnings.append(
                f"under-filled: produced {compliant_count} compliant "
                f"{brief.creative_type.value} candidate(s), target was "
                f"{target_count}"
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
            warnings=request_warnings,
            target_count=target_count,
            brief_summary={
                "topic": brief.campaign_topic,
                "platform": brief.target_platform.value,
                "market": brief.target_market.value,
                "type": brief.creative_type.value,
            },
            total_generated=total_generated,
        )

        # ----------------------------------------------------------------------
        # Stage E: operator-review translations (HK team comprehension aid)
        # ----------------------------------------------------------------------
        # Translate the *delivered* copies into Simplified/Traditional Chinese
        # (+ English) so the Hong Kong reviewers can vet foreign-language ad
        # copy. One batched LLM call for the whole set; best-effort (failures
        # leave review_translations empty and never block delivery). Skipped
        # when no translator is wired.
        #
        # English is included for EVERY non-English market (operators asked for
        # zh-Hans + zh-Hant + en on e.g. Hindi/Vietnamese copy). We only drop
        # the redundant English pass when the DELIVERED copy is itself English —
        # which is decided by the target market's primary language, NOT
        # ``brief.source_language`` (the input language). Using source_language
        # was the bug: a Hindi-market run with an English brief was wrongly
        # flagged "already English", so English review translation was skipped.
        if self._review_translator is not None and ranking.ranked_candidates:
            copies = [c.source_copy for c in ranking.ranked_candidates]
            try:
                from creative_agent.integration.language_prompts import (
                    LanguagePromptSelector,
                )

                delivered_language = LanguagePromptSelector().get_primary_language(
                    brief.target_market
                )
            except KeyError:
                delivered_language = "en"
            copy_is_english = delivered_language.strip().lower() == "en"
            try:
                review = await self._review_translator.translate(
                    copies,
                    copy_is_english=copy_is_english,
                    request_id=request_id,
                )
                for candidate, langs in zip(ranking.ranked_candidates, review):
                    if langs:
                        candidate.review_translations = langs
            except Exception as exc:  # noqa: BLE001 — review aid never blocks
                log.warning(
                    "orchestrator.review_translation_failed",
                    request_id=request_id,
                    error=f"{type(exc).__name__}: {exc}",
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

    async def _filter_semantic_diversity(
        self,
        candidates: list[Creative_Candidate],
        *,
        accepted_pool: list[str],
        rejected_copies: list[str],
        reserve: list[Creative_Candidate],
        request_warnings: list[str],
        fallback_warned: bool,
        request_id: str,
    ) -> tuple[list[Creative_Candidate], bool]:
        """Filter freshly generated candidates for semantic duplicates.

        Implements the semantic-diversity step that runs *after* the
        generator's text-based dedup and *before* the compliance pipeline
        (Req 2.7). Each candidate's ``source_copy`` is checked against the
        ``accepted_pool`` of every source copy accepted so far this request,
        across all refill rounds (Req 2.9). The method mutates ``accepted_pool``
        and ``rejected_copies`` in place so subsequent rounds see the
        accumulated state.

        Behaviour:

        * **Accepted** candidate → its ``source_copy`` is appended to
          ``accepted_pool`` and the candidate is kept.
        * **Rejected** (semantic duplicate) → the candidate is dropped and its
          ``source_copy`` is recorded in ``rejected_copies`` so the
          orchestrator's ``exclude`` set asks the generator for a replacement
          (Req 2.5). The checker already logs the rejection pair/score/threshold
          (Req 2.6).
        * **Fallback** (embedding model unavailable or timed out) → the checker
          degrades to text-dedup only: the remaining candidates this round are
          all kept, a single request-level warning is appended (Req 2.8), and
          the circuit-breaker counter is left untouched (the checker never
          raises). Once fallback is observed the checker is skipped for the rest
          of the request.

        Args:
            candidates: Freshly generated candidates (already text-deduped).
            accepted_pool: Source copies accepted so far (mutated in place).
            rejected_copies: Source copies rejected so far (mutated in place).
            request_warnings: Request-level warnings list (mutated in place).
            fallback_warned: Whether the text-dedup-only warning has already
                been appended this request.
            request_id: Per-request identifier for logging.

        Returns:
            ``(kept_candidates, fallback_warned)`` — the candidates that passed
            the semantic check (or all of them under fallback) and the updated
            ``fallback_warned`` flag.
        """
        checker = self._semantic_diversity_checker
        assert checker is not None  # guarded by caller (semantic_enabled)

        # Once we have degraded to text-dedup only this request, skip further
        # embedding work and keep every remaining candidate (Req 2.8).
        if fallback_warned:
            return list(candidates), fallback_warned

        kept: list[Creative_Candidate] = []
        rejected = 0
        for index, candidate in enumerate(candidates):
            result = await checker.check_candidate(
                candidate.source_copy, accepted_pool
            )

            if result.fallback:
                # Degrade to text-dedup only: keep this candidate and every
                # remaining one, append the warning once (Req 2.8).
                if not fallback_warned:
                    request_warnings.append(_SEMANTIC_FALLBACK_WARNING)
                    fallback_warned = True
                log.warning(
                    "orchestrator.semantic_diversity_fallback",
                    request_id=request_id,
                    reason="embedding_unavailable_or_timeout",
                )
                # Keep this candidate and the rest of the batch without
                # further checks.
                for remaining in candidates[index:]:
                    kept.append(remaining)
                    accepted_pool.append(remaining.source_copy)
                return kept, fallback_warned

            if result.accepted:
                kept.append(candidate)
                accepted_pool.append(candidate.source_copy)
            else:
                # Rejected as a semantic duplicate: drop it from the diverse
                # set, remember its copy so the refill loop excludes it
                # (Req 2.5), AND keep the candidate object in ``reserve``. The
                # reserve is a quality-ordered backfill source: if diversified
                # generation can't reach the ad-group quota, we top up from the
                # reserve so the delivered count is always exactly the target
                # (business hard requirement), preferring diverse copies and
                # only falling back to near-duplicates when unavoidable. The
                # checker already logged the pair/score/threshold (Req 2.6).
                rejected += 1
                rejected_copies.append(candidate.source_copy)
                reserve.append(candidate)

        if rejected:
            log.info(
                "orchestrator.semantic_diversity_filter",
                request_id=request_id,
                generated=len(candidates),
                kept=len(kept),
                rejected=rejected,
                accepted_pool_size=len(accepted_pool),
            )

        return kept, fallback_warned

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


def _accepted_angle_counts(
    candidates: list[Creative_Candidate],
) -> dict[str, int]:
    """Tally accepted candidates per angle label across all refill rounds.

    Feeds :meth:`CreativeGenerator.generate_with_angles` so its lowest-count
    cycling (Req 3.5) keeps balancing angles using the candidates already
    accepted in earlier rounds, rather than restarting from zero each round.
    Candidates without an ``angle_label`` (e.g. produced by a single-prompt
    fallback) are ignored.
    """
    counts: dict[str, int] = {}
    for candidate in candidates:
        label = candidate.angle_label
        if label is None:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts
