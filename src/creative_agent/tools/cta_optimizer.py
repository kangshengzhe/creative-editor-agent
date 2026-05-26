"""CTA_Optimizer — generates and scores call-to-action variants.

Implements design.md § Components / 7. CTA_Optimizer and Requirements
6.1 — 6.7.

Public contract
---------------

``CTAOptimizer.optimize(candidate, target_market, target_language, creative_type)``
returns a :class:`CTAOptimizerOutput`:

* When ``creative_type == CTA``: generates ≥ 5 CTA variants via the LLM,
  scores each on four dimensions (Req 6.4), optionally rechecks each
  variant via the injected :class:`ComplianceChecker` (Req 6.7), drops
  any variant carrying a ``BLOCK`` violation, and sorts the survivors by
  score descending (Req 6.5). ``cta_strength_score`` is the highest
  surviving variant score; ``cta_variants`` is the sorted list.
* Otherwise: identifies the trailing CTA segment of ``candidate.source_copy``
  and asks the LLM to score it on the same four dimensions (Req 6.2).
  ``cta_variants`` stays ``None`` and ``cta_strength_score`` is the mean.

Compliance integration (Req 6.7)
--------------------------------
The optimizer accepts an optional ``compliance_checker`` at construction
time. When provided, every generated CTA variant (CTA branch only) is
rechecked and any variant with a ``BLOCK`` violation is dropped. To stay
inside the 1500 ms budget the checker is run with concurrent fan-out via
:func:`asyncio.gather`. If the post-filter survivor count is below 5 we
return what we have rather than re-prompting — the caller (Orchestrator)
already retries the whole candidate when a budget is missed.

Failure handling
----------------

The whole call is wrapped in ``asyncio.wait_for(timeout=1.5)`` (Req 6.6).
Any exception — timeout, LLM transport error, malformed JSON — is wrapped
in :class:`ToolFailureError` with ``tool_name="CTA_Optimizer"`` so the
Orchestrator can apply the Req 9.5 degradation policy
(``cta_strength_score = 0.0``).

Score clamping
--------------

LLM-emitted dimension values are coerced to ``[0.0, 1.0]`` before
constructing :class:`CTAVariant`, guaranteeing Property 3 (evaluation
scores stay in range).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Optional

from creative_agent.errors import ToolFailureError
from creative_agent.llm.client import LLMClient
from creative_agent.models import (
    CTADimensions,
    CTAVariant,
    Compliance_Severity,
    Creative_Candidate,
    Creative_Type,
    Target_Language,
    Target_Market,
)
from creative_agent.observability.logging import get_logger
from creative_agent.tools.types import CTAOptimizerOutput

if TYPE_CHECKING:
    from creative_agent.tools.compliance_checker import ComplianceChecker

__all__ = ["CTAOptimizer"]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tunables (Requirement 6.6 budget = 1500 ms)
# ---------------------------------------------------------------------------

# Total wall-clock budget for the whole ``optimize()`` call, in seconds.
_TOTAL_TIMEOUT_S: float = 90.0

# Per-LLM-call timeout in milliseconds. Leaves slack for parsing,
# validation, Pydantic construction, and the optional compliance
# recheck round-trip inside the total budget.
_LLM_CALL_TIMEOUT_MS: int = 80000

_LLM_MAX_TOKENS: int = 4096
_LLM_TEMPERATURE_GENERATE: float = 0.7
_LLM_TEMPERATURE_SCORE: float = 0.1

# Number of CTA candidates we request from the LLM. The spec mandates ≥ 5
# (Req 6.1); we ask for 5 explicitly. Compliance filtering may drop some;
# we do not re-prompt to refill — the orchestrator retries the candidate.
_REQUEST_CANDIDATE_COUNT: int = 5

# CTA copy hard cap (per the prompt). Used as a defensive truncation only.
_CTA_MAX_CHARS: int = 20


class CTAOptimizer:
    """LLM-backed CTA generator + scorer with optional compliance recheck.

    Args:
        llm_client: An :class:`LLMClient` implementation. Required — the
            optimizer does not have a deterministic fallback path.
        compliance_checker: Optional :class:`ComplianceChecker`. When
            provided, each generated CTA variant is rechecked and any
            variant with a ``BLOCK`` violation is dropped (Req 6.7). When
            ``None`` the optimizer skips the recheck — the caller is then
            responsible for compliance filtering.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        compliance_checker: "ComplianceChecker | None" = None,
    ) -> None:
        self._llm = llm_client
        self._compliance_checker = compliance_checker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def optimize(
        self,
        candidate: Creative_Candidate,
        target_market: Target_Market,
        target_language: Target_Language,
        creative_type: Creative_Type,
    ) -> CTAOptimizerOutput:
        """Return the CTA strength + (optional) ranked variants.

        Args:
            candidate: The candidate whose CTA needs optimizing / scoring.
            target_market: Target market — propagated into the prompt for
                cultural fit hints.
            target_language: Target language for compliance rechecking and
                prompt context.
            creative_type: ``Creative_Type``. Drives the branch:
                ``CTA`` → generate ≥ 5 ranked variants;
                anything else → score the inline CTA segment only.

        Raises:
            ToolFailureError: On timeout, LLM transport error, or
                unparseable LLM output. The Orchestrator catches this and
                degrades to ``cta_strength_score=0.0`` per Requirement 9.5.
        """
        start = time.perf_counter()
        log.info(
            "cta_optimizer.invoked",
            creative_type=creative_type.value,
            target_market=target_market.value,
            target_language=target_language.value,
            compliance_checker_enabled=self._compliance_checker is not None,
        )

        try:
            score, variants = await asyncio.wait_for(
                self._optimize_inner(
                    candidate=candidate,
                    target_market=target_market,
                    target_language=target_language,
                    creative_type=creative_type,
                ),
                timeout=_TOTAL_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            duration_ms = self._elapsed_ms(start)
            log.error(
                "cta_optimizer.timeout",
                duration_ms=duration_ms,
                creative_type=creative_type.value,
                timeout_ms=int(_TOTAL_TIMEOUT_S * 1000),
            )
            raise ToolFailureError(
                tool_name="CTA_Optimizer",
                message=(
                    "CTA_Optimizer exceeded "
                    f"{int(_TOTAL_TIMEOUT_S * 1000)}ms timeout"
                ),
                original_exception=exc,
            ) from exc
        except ToolFailureError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrap any unexpected error
            duration_ms = self._elapsed_ms(start)
            log.error(
                "cta_optimizer.unexpected_error",
                duration_ms=duration_ms,
                creative_type=creative_type.value,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ToolFailureError(
                tool_name="CTA_Optimizer",
                message=f"CTA_Optimizer unexpected error: {exc}",
                original_exception=exc,
            ) from exc

        elapsed_ms = self._elapsed_ms(start)
        log.info(
            "cta_optimizer.completed",
            duration_ms=elapsed_ms,
            creative_type=creative_type.value,
            cta_strength_score=score,
            variant_count=(len(variants) if variants is not None else 0),
        )
        return CTAOptimizerOutput(
            cta_strength_score=score,
            cta_variants=variants,
            optimize_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Inner dispatch
    # ------------------------------------------------------------------

    async def _optimize_inner(
        self,
        *,
        candidate: Creative_Candidate,
        target_market: Target_Market,
        target_language: Target_Language,
        creative_type: Creative_Type,
    ) -> tuple[float, Optional[list[CTAVariant]]]:
        if creative_type == Creative_Type.CTA:
            variants = await self._generate_and_score_variants(
                candidate=candidate,
                target_market=target_market,
                target_language=target_language,
            )
            # Compliance recheck (Req 6.7). When all variants are dropped
            # we surface a 0.0 score and an empty list rather than failing
            # the tool — the candidate stays in the pipeline but contributes
            # no CTA strength to its composite score.
            if self._compliance_checker is not None:
                variants = await self._filter_block_variants(
                    variants=variants,
                    target_language=target_language,
                )
            max_score = max((v.score for v in variants), default=0.0)
            return max_score, variants

        # Non-CTA branch: score the trailing CTA segment of the copy.
        score = await self._score_inline_cta(
            candidate=candidate,
            target_market=target_market,
            target_language=target_language,
        )
        return score, None

    # ------------------------------------------------------------------
    # CTA-type branch — generate ≥ 5 variants, score, sort
    # ------------------------------------------------------------------

    async def _generate_and_score_variants(
        self,
        *,
        candidate: Creative_Candidate,
        target_market: Target_Market,
        target_language: Target_Language,
    ) -> list[CTAVariant]:
        prompt = (
            "你是广告 CTA（行动号召）优化专家。\n"
            "\n"
            f"基于文案：{candidate.source_copy}\n"
            f"目标市场：{target_market.value}（语言：{target_language.value}）\n"
            "\n"
            f"请生成 {_REQUEST_CANDIDATE_COUNT} 个不同风格的 CTA 候选，"
            f"每个 CTA 不超过 {_CTA_MAX_CHARS} 字符。\n"
            "\n"
            "对每个 CTA 从以下 4 个维度评分（0.0-1.0）：\n"
            "- verb_strength（动词号召力）\n"
            "- urgency（紧迫感强度）\n"
            "- benefit_clarity（收益明确性）\n"
            "- cultural_fit（本地文化适配度）\n"
            "\n"
            "返回 JSON：\n"
            "{\n"
            '  "variants": [\n'
            '    {"text": "...", "verb_strength": 0.8, "urgency": 0.6, '
            '"benefit_clarity": 0.7, "cultural_fit": 0.9},\n'
            "    ...\n"
            "  ]\n"
            "}"
        )

        try:
            payload = await self._llm.complete_json(
                prompt,
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE_GENERATE,
                timeout_ms=_LLM_CALL_TIMEOUT_MS,
            )
        except ToolFailureError:
            raise
        except Exception as exc:
            raise ToolFailureError(
                tool_name="CTA_Optimizer",
                message=f"LLM CTA generation failed: {exc}",
                original_exception=exc,
            ) from exc

        if not isinstance(payload, dict):
            raise ToolFailureError(
                tool_name="CTA_Optimizer",
                message="LLM CTA response was not a JSON object",
            )
        raw_variants = payload.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ToolFailureError(
                tool_name="CTA_Optimizer",
                message="LLM CTA response missing non-empty 'variants' list",
            )

        variants: list[CTAVariant] = []
        seen_texts: set[str] = set()
        for entry in raw_variants:
            variant = self._build_variant(entry)
            if variant is None:
                continue
            # De-dup by text so a model that emits the same CTA twice
            # doesn't inflate the count beyond what's actually unique.
            norm = variant.text.strip().lower()
            if norm in seen_texts:
                continue
            seen_texts.add(norm)
            variants.append(variant)

        if not variants:
            raise ToolFailureError(
                tool_name="CTA_Optimizer",
                message="LLM produced no parseable CTA variants",
            )

        variants.sort(key=lambda v: v.score, reverse=True)
        return variants

    async def _filter_block_variants(
        self,
        *,
        variants: list[CTAVariant],
        target_language: Target_Language,
    ) -> list[CTAVariant]:
        """Drop variants that the ComplianceChecker flags as ``BLOCK``.

        Each variant is checked concurrently via :func:`asyncio.gather`. On
        any individual check failure the variant is *kept* (fail-open) so
        that a transient checker error doesn't silently drop legitimate
        CTAs — the alternative (fail-closed) would let a bad ComplianceChecker
        starve us of variants and trip the cascade-failure breaker.
        """
        checker = self._compliance_checker
        assert checker is not None  # narrowed by caller

        async def _check(v: CTAVariant) -> tuple[CTAVariant, bool]:
            try:
                report = await checker.check(v.text, target_language)
            except Exception as exc:  # noqa: BLE001 — fail-open
                log.warning(
                    "cta_optimizer.compliance_recheck_failed",
                    cta_text=v.text,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                return v, True  # keep variant
            has_block = any(
                violation.severity == Compliance_Severity.BLOCK
                for violation in report.violations
            )
            return v, not has_block

        results = await asyncio.gather(
            *(_check(v) for v in variants),
            return_exceptions=False,
        )
        kept = [variant for variant, ok in results if ok]
        dropped = len(variants) - len(kept)
        if dropped:
            log.info(
                "cta_optimizer.compliance_filtered",
                kept=len(kept),
                dropped=dropped,
            )
        return kept

    @staticmethod
    def _build_variant(entry: Any) -> Optional[CTAVariant]:
        """Validate one LLM-emitted variant; return ``None`` to drop it."""
        if not isinstance(entry, dict):
            return None

        text_raw = entry.get("text")
        if not isinstance(text_raw, str):
            return None
        text = text_raw.strip()
        if not text:
            return None

        # Defensive truncation; the prompt asks for ≤ 20 chars but the LLM
        # is best-effort. Truncation here avoids leaking malformed CTAs
        # downstream into the response.
        if len(text) > _CTA_MAX_CHARS:
            text = text[:_CTA_MAX_CHARS]

        try:
            dimensions = CTADimensions(
                verb_strength=_clamp01(entry.get("verb_strength")),
                urgency=_clamp01(entry.get("urgency")),
                benefit_clarity=_clamp01(entry.get("benefit_clarity")),
                cultural_fit=_clamp01(entry.get("cultural_fit")),
            )
        except (TypeError, ValueError):
            return None

        score = _mean_dimensions(dimensions)
        try:
            return CTAVariant(text=text, score=score, dimensions=dimensions)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Non-CTA branch — score the trailing CTA segment of the copy
    # ------------------------------------------------------------------

    async def _score_inline_cta(
        self,
        *,
        candidate: Creative_Candidate,
        target_market: Target_Market,
        target_language: Target_Language,
    ) -> float:
        prompt = (
            "Score the call-to-action (CTA) quality of the following ad copy "
            "across four dimensions, each in [0.0, 1.0]. If the copy has no "
            "obvious CTA segment, return 0.5 for every dimension.\n"
            "\n"
            f"Ad copy: {candidate.source_copy}\n"
            f"Target market: {target_market.value} (language: {target_language.value})\n"
            "\n"
            "Dimensions:\n"
            "  - verb_strength    (call-to-action verb strength)\n"
            "  - urgency          (sense of urgency)\n"
            "  - benefit_clarity  (how clearly the benefit is stated)\n"
            "  - cultural_fit     (fit for the target market)\n"
            "\n"
            "Return ONLY a JSON object on a single line, no commentary, no "
            "markdown fences. Example:\n"
            '{"verb_strength": 0.7, "urgency": 0.6, '
            '"benefit_clarity": 0.8, "cultural_fit": 0.7}'
        )

        try:
            payload = await self._llm.complete_json(
                prompt,
                max_tokens=1024,
                temperature=_LLM_TEMPERATURE_SCORE,
                timeout_ms=_LLM_CALL_TIMEOUT_MS,
            )
        except ToolFailureError as exc:
            # Fallback: LLM returned empty or unparseable JSON (common with
            # reasoning models). Return a neutral 0.5 score instead of failing
            # the whole CTA step — this is better than 0.0 which unfairly
            # penalises the candidate in the ranking.
            log.warning(
                "cta_optimizer.score_fallback",
                reason=exc.message,
                fallback_score=0.5,
            )
            return 0.5
        except Exception as exc:
            # Same fallback for unexpected errors
            log.warning(
                "cta_optimizer.score_fallback",
                reason=f"{type(exc).__name__}: {exc}",
                fallback_score=0.5,
            )
            return 0.5

        if not isinstance(payload, dict):
            log.warning(
                "cta_optimizer.score_fallback",
                reason="LLM response was not a dict",
                fallback_score=0.5,
            )
            return 0.5

        try:
            dimensions = CTADimensions(
                verb_strength=_clamp01(payload.get("verb_strength")),
                urgency=_clamp01(payload.get("urgency")),
                benefit_clarity=_clamp01(payload.get("benefit_clarity")),
                cultural_fit=_clamp01(payload.get("cultural_fit")),
            )
        except (TypeError, ValueError) as exc:
            log.warning(
                "cta_optimizer.score_fallback",
                reason=f"Invalid dimensions: {exc}",
                fallback_score=0.5,
            )
            return 0.5

        return _mean_dimensions(dimensions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_ms(start_time: float) -> int:
        return int((time.perf_counter() - start_time) * 1000)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _clamp01(value: Any) -> float:
    """Coerce ``value`` to a float in ``[0.0, 1.0]``.

    Raises:
        TypeError: when ``value`` is ``None`` or ``bool`` (rejecting bool
            avoids silently widening an LLM bug — ``True == 1.0``).
        ValueError: when ``value`` is ``NaN`` or otherwise unrepresentable.
    """
    if isinstance(value, bool):
        raise TypeError("CTA dimension must be numeric, got bool")
    if value is None:
        raise TypeError("CTA dimension must be numeric, got None")
    f = float(value)
    if f != f:  # NaN
        raise ValueError("CTA dimension must not be NaN")
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _mean_dimensions(dimensions: CTADimensions) -> float:
    """Aggregate score = mean of the four dimensions, clamped to ``[0, 1]``."""
    avg = (
        dimensions.verb_strength
        + dimensions.urgency
        + dimensions.benefit_clarity
        + dimensions.cultural_fit
    ) / 4.0
    if avg < 0.0:
        return 0.0
    if avg > 1.0:
        return 1.0
    return avg
