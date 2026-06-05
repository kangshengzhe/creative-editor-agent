"""Per-candidate processing pipeline.

Implements design.md § Architecture / "Pipeline" — the sequence of tool
invocations that takes a freshly generated :class:`Creative_Candidate` from
:class:`creative_agent.tools.CreativeGenerator` and turns it into a fully
populated candidate ready to be ranked by Composite Scorer.

Pipeline order (design.md § Components and Interfaces):

1. **Compliance_Checker (initial check)** on ``candidate.source_copy`` in the
   source language (Requirement 3.1 / 3.2). A BLOCK verdict does *not*
   short-circuit the pipeline — the BLOCK information is preserved on the
   candidate and the actual filtering is done by Composite Scorer at
   ranking time (Requirement 7.5).
2. **Localization_Tool** — translate the source copy into every language
   the target market requires (Requirements 4.1 — 4.4). Per-language
   failures already land in ``LocalizerOutput.failed_languages`` so we
   simply forward both fields onto the candidate (Requirement 9.3).
3. **Keyword_Embedder** — embed brief.keywords into the source copy. The
   embedded copy *replaces* ``candidate.source_copy`` so the recheck and
   CTA optimizer both run on the final wording (Requirements 5.1 — 5.9).
4. **Compliance_Checker (recheck)** on the post-embed copy. Embedding can
   introduce new keywords that themselves trigger violations, so we must
   re-run compliance after the rewrite (design.md § Architecture).
5. **CTA_Optimizer** — score the trailing CTA segment, or generate ranked
   CTA variants when ``creative_type == CTA`` (Requirements 6.1 — 6.7).

Tool-level degradation policy (Requirements 9.2 — 9.5):

* Compliance_Checker degrades **internally** by returning a Compliance_Report
  with a single WARN entry; it never raises ``ToolFailureError`` for
  business-level failures. We still detect that degradation and append a
  ``compliance_check_failed`` warning so the candidate can be flagged for
  human review.
* Localization_Tool may raise ``ToolFailureError`` only on a hard timeout;
  per-language failures are inside ``LocalizerOutput.failed_languages`` and
  are never fatal. A ``ToolFailureError`` here adds a warning and leaves
  ``localized_versions`` / ``failed_languages`` untouched.
* Keyword_Embedder raises ``ToolFailureError`` on any unrecoverable failure;
  we set ``keyword_coverage = 0.0`` and warn. The original copy is kept.
* CTA_Optimizer raises ``ToolFailureError`` on any unrecoverable failure;
  we set ``cta_strength_score = 0.0`` and warn.

The shared ``tool_failure_counter[0]`` is incremented on every tool failure
that the orchestrator's global circuit-breaker should observe
(Requirement 9.6). Compliance_Checker's *internal* degradation does **not**
count, because the tool returns a valid (degraded) report instead of
raising — that is exactly the "graceful degradation" the spec asks for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from creative_agent.errors import ToolFailureError
from creative_agent.integration.display_width import DisplayWidthCalculator
from creative_agent.integration.language_prompts import LanguagePromptSelector
from creative_agent.models import (
    Creative_Brief,
    Creative_Candidate,
    Target_Language,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.observability.logging import get_logger
from creative_agent.tools.localization_tool import market_to_languages

if TYPE_CHECKING:
    from creative_agent.tools.compliance_checker import ComplianceChecker
    from creative_agent.tools.cta_optimizer import CTAOptimizer
    from creative_agent.tools.keyword_embedder import KeywordEmbedder
    from creative_agent.tools.localization_tool import LocalizationTool

__all__ = ["PipelineDeps", "process_candidate"]

log = get_logger(__name__)

# Stateless selector used to resolve a market's primary language so we can
# skip redundant translation when the brief's source language already matches
# it (Requirement 1.4). Held as a module-level singleton because it carries no
# per-request state.
_LANGUAGE_PROMPT_SELECTOR = LanguagePromptSelector()

# Stateless Display_Width_Calculator singleton (Req 4.8 / 4.10). Used to keep a
# candidate's ``display_width`` in sync with its ``source_copy`` after the
# keyword-embed stage rewrites the copy — for CJK copy the rewrite changes the
# Display_Unit count, so the value the generator stamped at construction time
# would otherwise be stale.
_DISPLAY_WIDTH = DisplayWidthCalculator()


# Allow-list of language codes the source can carry without forcing us to
# coerce to ``EN``. Anything else falls back to ``EN`` so downstream tools
# (which only accept ``Target_Language``) never see a surprise value.
_SUPPORTED_SOURCE_LANGS: frozenset[str] = frozenset({"en", "fil", "th", "ru"})


# ---------------------------------------------------------------------------
# Dependency container
# ---------------------------------------------------------------------------


@dataclass
class PipelineDeps:
    """Pipeline dependencies injected by the orchestrator.

    Attributes:
        compliance_checker: Shared Compliance_Checker instance used for both
            the initial check and the post-embed recheck.
        localization_tool: Shared Localization_Tool instance.
        keyword_embedder: Shared Keyword_Embedder instance.
        cta_optimizer: Shared CTA_Optimizer instance.
        request_id: Request id used for structured logging. Optional so the
            pipeline is testable in isolation.
        tool_failure_counter: Single-element list serving as a shared mutable
            counter for the global circuit breaker (Requirement 9.6). The
            orchestrator owns the list; pipeline only increments index 0
            when a tool call fails. ``None`` disables counting (useful for
            unit tests that don't care about the breaker).
    """

    compliance_checker: "ComplianceChecker"
    localization_tool: "LocalizationTool"
    keyword_embedder: "KeywordEmbedder"
    cta_optimizer: "CTAOptimizer"
    request_id: Optional[str] = None
    tool_failure_counter: Optional[list[int]] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def process_candidate(
    candidate: Creative_Candidate,
    brief: Creative_Brief,
    platform_spec: Platform_Spec,
    deps: PipelineDeps,
) -> Creative_Candidate:
    """Run the full per-candidate processing pipeline.

    Returns the same ``candidate`` object with its tool-output fields
    populated (``compliance_report``, ``localized_versions``,
    ``failed_languages``, ``keyword_coverage``, ``hit_keywords``,
    ``skipped_keywords``, ``source_copy``, ``cta_strength_score``,
    ``cta_variants``, and any ``warnings`` accumulated along the way).

    The function never raises. Every tool-level failure is captured into
    ``candidate.warnings`` and (for failures that bypass a tool's internal
    degradation) bumps ``deps.tool_failure_counter[0]``.
    """
    candidate_id = candidate.candidate_id
    source_lang = _resolve_source_language(brief.source_language)

    # -- Stage 1: Compliance initial check -------------------------------
    initial_report = await _run_initial_compliance(
        candidate=candidate,
        deps=deps,
        candidate_id=candidate_id,
        source_lang=source_lang,
    )
    candidate.compliance_report = initial_report

    # -- Stage 2: Localization -------------------------------------------
    await _run_localization(
        candidate=candidate,
        brief=brief,
        deps=deps,
        candidate_id=candidate_id,
        source_lang=source_lang,
    )

    # -- Stage 3: Keyword embedding --------------------------------------
    await _run_keyword_embed(
        candidate=candidate,
        brief=brief,
        platform_spec=platform_spec,
        deps=deps,
        candidate_id=candidate_id,
    )

    # -- Stage 4: Compliance recheck on post-embed copy -------------------
    recheck_report = await _run_compliance_recheck(
        candidate=candidate,
        deps=deps,
        candidate_id=candidate_id,
        source_lang=source_lang,
    )
    candidate.compliance_report = recheck_report

    # -- Stage 5: CTA optimization ---------------------------------------
    await _run_cta_optimize(
        candidate=candidate,
        brief=brief,
        deps=deps,
        candidate_id=candidate_id,
        source_lang=source_lang,
    )

    return candidate


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


async def _run_initial_compliance(
    *,
    candidate: Creative_Candidate,
    deps: PipelineDeps,
    candidate_id: str,
    source_lang: Target_Language,
) -> "object":
    """Run the first compliance check on the original source copy."""
    start = time.perf_counter()
    log.info(
        "pipeline.compliance_initial.start",
        request_id=deps.request_id,
        candidate_id=candidate_id,
        tool_name="Compliance_Checker",
    )
    # Compliance_Checker.check is documented to never raise — it returns a
    # degraded WARN report on any internal failure. Belt-and-braces try
    # remains here so a misbehaving subclass cannot poison the pipeline.
    try:
        report = await deps.compliance_checker.check(
            candidate.source_copy, source_lang
        )
    except Exception as exc:  # noqa: BLE001 — defensive
        duration_ms = _elapsed_ms(start)
        _bump_failure_counter(deps.tool_failure_counter)
        msg = f"compliance_check_failed: {type(exc).__name__}: {exc}"
        candidate.warnings = [*candidate.warnings, msg]
        log.warning(
            "pipeline.compliance_initial.error",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Compliance_Checker",
            status="ERROR",
            duration_ms=duration_ms,
            error=str(exc),
        )
        # Fall back to the existing (placeholder) report so the candidate
        # still has a well-formed Compliance_Report.
        return candidate.compliance_report

    duration_ms = _elapsed_ms(start)
    # Detect *internal* degradation: the tool advertises this as a single
    # WARN entry whose suggestion starts with "合规检查失败" (see
    # ComplianceChecker._degraded_report).
    if _looks_degraded(report):
        candidate.warnings = [
            *candidate.warnings,
            "compliance_check_failed: degraded report (initial check)",
        ]
        log.warning(
            "pipeline.compliance_initial.degraded",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Compliance_Checker",
            status="DEGRADED",
            duration_ms=duration_ms,
        )
    else:
        log.info(
            "pipeline.compliance_initial.done",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Compliance_Checker",
            status="OK",
            duration_ms=duration_ms,
            score=report.compliance_score,
            violation_count=len(report.violations),
        )
    return report


async def _run_localization(
    *,
    candidate: Creative_Candidate,
    brief: Creative_Brief,
    deps: PipelineDeps,
    candidate_id: str,
    source_lang: Target_Language,
) -> None:
    """Translate the candidate into every language the target market needs.

    When ``brief.source_language`` matches the target market's primary
    language, the translate step for that language is skipped — the source
    copy is already native in that language, so re-translating it would be
    redundant (Requirement 1.4). The remaining market languages are still
    translated. When *no* languages remain after the skip (e.g. an
    English-only market with an English source), the translate call is
    bypassed entirely.
    """
    start = time.perf_counter()

    # Decide which languages still need translation after removing any
    # language that is redundant with the source language (Req 1.4).
    target_languages, skipped_language = _resolve_target_languages(brief=brief)

    if skipped_language is not None:
        log.info(
            "pipeline.localization.skip_redundant",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Localization_Tool",
            target_market=brief.target_market.value,
            language=skipped_language,
        )

    # Nothing left to translate — skip the tool call entirely (Req 1.4).
    if target_languages is not None and not target_languages:
        candidate.localized_versions = {}
        candidate.failed_languages = []
        log.info(
            "pipeline.localization.skipped",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Localization_Tool",
            target_market=brief.target_market.value,
            reason="source_language matches the market's only language",
        )
        return

    log.info(
        "pipeline.localization.start",
        request_id=deps.request_id,
        candidate_id=candidate_id,
        tool_name="Localization_Tool",
        target_market=brief.target_market.value,
    )
    try:
        result = await deps.localization_tool.translate(
            candidate.source_copy,
            source_language=source_lang,
            target_languages=target_languages,  # None ⇒ derive from market
            target_market=brief.target_market,
        )
    except ToolFailureError as exc:
        duration_ms = _elapsed_ms(start)
        _bump_failure_counter(deps.tool_failure_counter)
        candidate.warnings = [
            *candidate.warnings,
            f"localization_failed: {exc.message}",
        ]
        log.warning(
            "pipeline.localization.error",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Localization_Tool",
            status="ERROR",
            duration_ms=duration_ms,
            error=exc.message,
        )
        return
    except Exception as exc:  # noqa: BLE001 — defensive
        duration_ms = _elapsed_ms(start)
        _bump_failure_counter(deps.tool_failure_counter)
        candidate.warnings = [
            *candidate.warnings,
            f"localization_failed: {type(exc).__name__}: {exc}",
        ]
        log.warning(
            "pipeline.localization.error",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Localization_Tool",
            status="ERROR",
            duration_ms=duration_ms,
            error=str(exc),
        )
        return

    duration_ms = _elapsed_ms(start)
    candidate.localized_versions = dict(result.localized_versions)
    candidate.failed_languages = list(result.failed_languages)
    log.info(
        "pipeline.localization.done",
        request_id=deps.request_id,
        candidate_id=candidate_id,
        tool_name="Localization_Tool",
        status="OK",
        duration_ms=duration_ms,
        translated_count=len(result.localized_versions),
        failed_count=len(result.failed_languages),
    )


async def _run_keyword_embed(
    *,
    candidate: Creative_Candidate,
    brief: Creative_Brief,
    platform_spec: Platform_Spec,
    deps: PipelineDeps,
    candidate_id: str,
) -> None:
    """Embed SEO keywords; replace ``source_copy`` with the embedded version."""
    start = time.perf_counter()
    keywords = list(brief.keywords or [])
    log.info(
        "pipeline.keyword_embed.start",
        request_id=deps.request_id,
        candidate_id=candidate_id,
        tool_name="Keyword_Embedder",
        keyword_count=len(keywords),
    )
    try:
        result = await deps.keyword_embedder.embed(
            candidate.source_copy,
            keywords,
            platform_spec,
            brief.creative_type,
            language=candidate.generation_language,
        )
    except ToolFailureError as exc:
        duration_ms = _elapsed_ms(start)
        _bump_failure_counter(deps.tool_failure_counter)
        candidate.keyword_coverage = 0.0
        candidate.hit_keywords = []
        candidate.skipped_keywords = list(keywords)
        candidate.warnings = [
            *candidate.warnings,
            f"keyword_embed_failed: {exc.message}",
        ]
        log.warning(
            "pipeline.keyword_embed.error",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Keyword_Embedder",
            status="ERROR",
            duration_ms=duration_ms,
            error=exc.message,
        )
        return
    except Exception as exc:  # noqa: BLE001 — defensive
        duration_ms = _elapsed_ms(start)
        _bump_failure_counter(deps.tool_failure_counter)
        candidate.keyword_coverage = 0.0
        candidate.hit_keywords = []
        candidate.skipped_keywords = list(keywords)
        candidate.warnings = [
            *candidate.warnings,
            f"keyword_embed_failed: {type(exc).__name__}: {exc}",
        ]
        log.warning(
            "pipeline.keyword_embed.error",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Keyword_Embedder",
            status="ERROR",
            duration_ms=duration_ms,
            error=str(exc),
        )
        return

    duration_ms = _elapsed_ms(start)
    # Replace source_copy with the embedded version so the recheck +
    # CTA_Optimizer run on the final wording (design.md § Architecture).
    candidate.source_copy = result.embedded_copy
    # The embedded copy may differ in length from the generated copy; refresh
    # the Display_Unit count so ``display_width`` stays accurate for CJK copy
    # whose Display_Unit count is not equal to ``len()`` (Req 4.8).
    candidate.display_width = _DISPLAY_WIDTH.text_width(result.embedded_copy)
    candidate.keyword_coverage = result.keyword_coverage
    candidate.hit_keywords = list(result.hit_keywords)
    candidate.skipped_keywords = list(result.skipped_keywords)
    if result.failure_reason:
        candidate.warnings = [
            *candidate.warnings,
            f"keyword_embed_partial: {result.failure_reason}",
        ]
    log.info(
        "pipeline.keyword_embed.done",
        request_id=deps.request_id,
        candidate_id=candidate_id,
        tool_name="Keyword_Embedder",
        status="OK",
        duration_ms=duration_ms,
        coverage=result.keyword_coverage,
        hit_count=len(result.hit_keywords),
        skipped_count=len(result.skipped_keywords),
    )


async def _run_compliance_recheck(
    *,
    candidate: Creative_Candidate,
    deps: PipelineDeps,
    candidate_id: str,
    source_lang: Target_Language,
) -> "object":
    """Re-run compliance on the post-embed copy (design.md § Architecture)."""
    start = time.perf_counter()
    log.info(
        "pipeline.compliance_recheck.start",
        request_id=deps.request_id,
        candidate_id=candidate_id,
        tool_name="Compliance_Checker",
    )
    try:
        report = await deps.compliance_checker.check(
            candidate.source_copy, source_lang
        )
    except Exception as exc:  # noqa: BLE001 — defensive
        duration_ms = _elapsed_ms(start)
        _bump_failure_counter(deps.tool_failure_counter)
        candidate.warnings = [
            *candidate.warnings,
            f"compliance_check_failed: {type(exc).__name__}: {exc}",
        ]
        log.warning(
            "pipeline.compliance_recheck.error",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Compliance_Checker",
            status="ERROR",
            duration_ms=duration_ms,
            error=str(exc),
        )
        # Keep whatever report we already have on the candidate.
        return candidate.compliance_report

    duration_ms = _elapsed_ms(start)
    if _looks_degraded(report):
        candidate.warnings = [
            *candidate.warnings,
            "compliance_check_failed: degraded report (recheck)",
        ]
        log.warning(
            "pipeline.compliance_recheck.degraded",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Compliance_Checker",
            status="DEGRADED",
            duration_ms=duration_ms,
        )
    else:
        log.info(
            "pipeline.compliance_recheck.done",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="Compliance_Checker",
            status="OK",
            duration_ms=duration_ms,
            score=report.compliance_score,
            violation_count=len(report.violations),
        )
    return report


async def _run_cta_optimize(
    *,
    candidate: Creative_Candidate,
    brief: Creative_Brief,
    deps: PipelineDeps,
    candidate_id: str,
    source_lang: Target_Language,
) -> None:
    """Generate / score CTA variants for the candidate."""
    start = time.perf_counter()
    log.info(
        "pipeline.cta_optimize.start",
        request_id=deps.request_id,
        candidate_id=candidate_id,
        tool_name="CTA_Optimizer",
        creative_type=brief.creative_type.value,
    )
    try:
        result = await deps.cta_optimizer.optimize(
            candidate,
            brief.target_market,
            source_lang,
            brief.creative_type,
        )
    except ToolFailureError as exc:
        duration_ms = _elapsed_ms(start)
        _bump_failure_counter(deps.tool_failure_counter)
        candidate.cta_strength_score = 0.0
        candidate.cta_variants = None
        candidate.warnings = [
            *candidate.warnings,
            f"cta_optimize_failed: {exc.message}",
        ]
        log.warning(
            "pipeline.cta_optimize.error",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="CTA_Optimizer",
            status="ERROR",
            duration_ms=duration_ms,
            error=exc.message,
        )
        return
    except Exception as exc:  # noqa: BLE001 — defensive
        duration_ms = _elapsed_ms(start)
        _bump_failure_counter(deps.tool_failure_counter)
        candidate.cta_strength_score = 0.0
        candidate.cta_variants = None
        candidate.warnings = [
            *candidate.warnings,
            f"cta_optimize_failed: {type(exc).__name__}: {exc}",
        ]
        log.warning(
            "pipeline.cta_optimize.error",
            request_id=deps.request_id,
            candidate_id=candidate_id,
            tool_name="CTA_Optimizer",
            status="ERROR",
            duration_ms=duration_ms,
            error=str(exc),
        )
        return

    duration_ms = _elapsed_ms(start)
    candidate.cta_strength_score = result.cta_strength_score
    candidate.cta_variants = (
        list(result.cta_variants) if result.cta_variants is not None else None
    )
    log.info(
        "pipeline.cta_optimize.done",
        request_id=deps.request_id,
        candidate_id=candidate_id,
        tool_name="CTA_Optimizer",
        status="OK",
        duration_ms=duration_ms,
        cta_strength_score=result.cta_strength_score,
        variant_count=(
            len(result.cta_variants) if result.cta_variants is not None else 0
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_source_language(value: object) -> Target_Language:
    """Coerce ``brief.source_language`` to a :class:`Target_Language`.

    Falls back to ``Target_Language.EN`` when the brief carries an unknown
    string (the Pydantic model leaves the field as a free-form ``str`` so
    callers can supply ``"en"`` without importing the enum).
    """
    if isinstance(value, Target_Language):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in _SUPPORTED_SOURCE_LANGS:
            try:
                return Target_Language(normalised)
            except ValueError:
                pass
    return Target_Language.EN


def _resolve_target_languages(
    *, brief: Creative_Brief
) -> tuple[Optional[list[Target_Language]], Optional[str]]:
    """Decide which languages the localization step should translate into.

    Implements the Requirement 1.4 skip: when ``brief.source_language`` equals
    the target market's primary language, the copy is already native in that
    language, so translating it again is redundant. Returns a
    ``(target_languages, skipped_language)`` tuple:

    * ``target_languages is None`` — pass ``None`` to
      ``LocalizationTool.translate`` so it derives the full language list from
      the market (the unchanged, pre-1.4 behaviour). Happens when the source
      language does *not* match the primary language, or when the market is
      unknown (defensive: never skip).
    * ``target_languages == []`` — every language the market needs *is* the
      source language; nothing remains to translate and the caller should
      skip the tool call entirely.
    * ``target_languages == [...]`` — the explicit remaining languages after
      removing the redundant source/primary language.

    ``skipped_language`` is the language code that was removed (for logging),
    or ``None`` when nothing was skipped.
    """
    try:
        primary = _LANGUAGE_PROMPT_SELECTOR.get_primary_language(brief.target_market)
    except KeyError:
        # Unknown market — do not skip; preserve existing behaviour (the 1.4
        # skip only triggers on a known primary-language match).
        return None, None

    source_lang = (brief.source_language or "").strip().lower()
    if not source_lang or source_lang != primary.strip().lower():
        # Source language does not match the market's primary language —
        # translate every market language exactly as before.
        return None, None

    # Source language matches the market's primary language: skip translating
    # into that language (Req 1.4) and translate only the remaining ones.
    try:
        market_languages = market_to_languages(brief.target_market)
    except KeyError:  # pragma: no cover — guarded by get_primary_language above
        return None, None

    remaining = [
        lang
        for lang in market_languages
        if lang.value.strip().lower() != source_lang
    ]
    return remaining, primary


def _looks_degraded(report: object) -> bool:
    """Heuristic: detect Compliance_Checker's internal degradation report.

    The checker emits a single WARN ``Violation`` whose ``suggestion`` starts
    with ``"合规检查失败"`` (see ``ComplianceChecker._degraded_report``). This
    helper does not raise on unexpected shapes; it merely returns ``False``
    when the report cannot be inspected, since the cost of a false negative
    is just a missing warning.
    """
    try:
        violations = getattr(report, "violations", None)
        if not violations or len(violations) != 1:
            return False
        only = violations[0]
        suggestion = getattr(only, "suggestion", "") or ""
        return suggestion.startswith("合规检查失败")
    except Exception:  # noqa: BLE001 — never let this helper raise
        return False


def _bump_failure_counter(counter: Optional[list[int]]) -> None:
    """Increment the orchestrator's shared tool-failure counter, if present."""
    if counter is not None and len(counter) > 0:
        counter[0] += 1


def _elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)
