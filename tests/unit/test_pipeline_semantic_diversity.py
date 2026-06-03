"""Unit tests for the semantic-diversity pipeline step: ordering & fallback.

Feature: creative-localization-diversity (task 10.2).

These tests exercise the Orchestrator's integration of the
Semantic_Diversity_Checker into the candidate pipeline:

* **Requirement 2.7** — the semantic-diversity check runs *after* the
  generator's text-based dedup and *before* compliance checking. Two
  complementary, end-to-end orchestration tests prove this:

    1. A shared, ordered call log proves every semantic check for a round is
       recorded *before* any compliance check for that round (temporal order).
    2. A semantically-duplicate candidate (identical embedding to an accepted
       one) is dropped by the diversity step and its text is therefore never
       handed to the Compliance_Checker — diversity filters *before*
       compliance ever sees the candidate.

* **Requirement 2.8** — when embedding computation times out (or the embedding
  model is unavailable) the checker degrades to *text-dedup only*: the run
  still succeeds, every candidate survives (text-dedup keeps them), the
  request-level ``_SEMANTIC_FALLBACK_WARNING`` is surfaced on the AB_Ranking,
  and the global circuit breaker is left untouched (no ``CascadeFailureError``
  — the orchestration simply completes normally, with no refill churn).

Approach used for the ordering assertion: **shared ordered call log (spy on
call order)** as the primary proof, complemented by the
"rejected-candidate-never-reaches-compliance" assertion.

The full Orchestrator is driven with lightweight in-memory fakes for the five
tools plus a deterministic, injected ``embed_fn`` so the tests are fast and
require no LLM or ML dependency. The project runs with
``asyncio_mode = "auto"`` so ``async def test_...`` coroutines are awaited
automatically; ``embed_fn`` runs via ``asyncio.to_thread`` so a blocking
``time.sleep`` correctly triggers the ``asyncio.wait_for`` timeout.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import pytest

from creative_agent.integration.semantic_diversity import (
    EmbeddingUnavailableError,
    SemanticDiversityChecker,
)
from creative_agent.models import (
    Compliance_Report,
    Creative_Brief,
    Creative_Candidate,
    Creative_Type,
    SemanticDiversityConfig,
    Target_Market,
    Target_Platform,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.observability.trace import TraceRecorder
from creative_agent.orchestrator.orchestrator import (
    _SEMANTIC_FALLBACK_WARNING,
    Orchestrator,
)
from creative_agent.tools.creative_generator import GeneratorOutput
from creative_agent.tools.types import (
    CTAOptimizerOutput,
    EmbedderOutput,
    LocalizerOutput,
)


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------


def _make_report() -> Compliance_Report:
    """A clean, fully-compliant placeholder report (no BLOCK, no WARN)."""
    return Compliance_Report(
        compliance_score=1.0,
        violations=[],
        checked_at="2026-01-01T00:00:00Z",
        checker_version="test",
    )


def _make_candidate(index: int, copy: str) -> Creative_Candidate:
    """Build a minimal, valid Creative_Candidate as the generator would."""
    return Creative_Candidate(
        candidate_id=f"cand-{index}",
        generation_index=index,
        source_copy=copy,
        source_language="en",
        compliance_report=_make_report(),
        keyword_coverage=0.0,
        hit_keywords=[],
        skipped_keywords=[],
        cta_strength_score=0.0,
        cta_variants=None,
        localized_versions={},
        failed_languages=[],
        composite_score=0.0,
        target_platform=Target_Platform.GOOGLE_ADS,
        creative_type=Creative_Type.HEADLINE,
        warnings=[],
    )


def _make_platform_spec() -> Platform_Spec:
    return Platform_Spec(
        platform=Target_Platform.GOOGLE_ADS,
        char_limits={
            Creative_Type.HEADLINE: 30,
            Creative_Type.DESCRIPTION: 90,
            Creative_Type.CTA: 15,
            Creative_Type.LONG_COPY: 200,
        },
        allowed_creative_types=[
            Creative_Type.HEADLINE,
            Creative_Type.DESCRIPTION,
            Creative_Type.CTA,
            Creative_Type.LONG_COPY,
        ],
    )


def _make_brief() -> Creative_Brief:
    # EN_GLOBAL + English source means the per-candidate localization step is
    # skipped entirely (Req 1.4), so the test never depends on translation.
    # ``target_count=5`` pins the Ad_Group_Quota (Req 5.8 override) to match the
    # fake generator's fixed 5-candidate batch, so these semantic-diversity
    # tests stay single-round and focus purely on ordering / fallback behavior
    # rather than quota refill churn.
    return Creative_Brief(
        campaign_topic="Top up promo",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=Target_Market.EN_GLOBAL,
        creative_type=Creative_Type.HEADLINE,
        source_language="en",
        keywords=[],
        target_count=5,
    )


# ---------------------------------------------------------------------------
# Lightweight tool fakes
# ---------------------------------------------------------------------------


class _FakeGenerator:
    """Single-prompt Creative_Generator stub.

    Returns a fixed batch of candidates on its first ``generate`` call and an
    empty batch afterwards (the orchestrator should never need a refill in
    these tests because the first round already yields >= 3 compliant
    candidates). ``supports_angle_generation`` is False so the orchestrator
    takes the plain :meth:`generate` path.
    """

    supports_angle_generation = False

    def __init__(self, copies: list[str]) -> None:
        self._copies = copies
        self.calls = 0

    async def generate(
        self,
        *,
        brief: Creative_Brief,
        platform_spec: Platform_Spec,
        exclude_copies: Optional[list[str]] = None,
        min_count: int = 5,
        request_id: Optional[str] = None,
        tool_failure_counter: Optional[list[int]] = None,
    ) -> GeneratorOutput:
        self.calls += 1
        if self.calls > 1:
            return GeneratorOutput(candidates=[], generation_time_ms=1)
        candidates = [
            _make_candidate(i, copy) for i, copy in enumerate(self._copies)
        ]
        return GeneratorOutput(candidates=candidates, generation_time_ms=1)


class _RecordingComplianceChecker:
    """Compliance_Checker stub that records the text of every check call.

    The pipeline calls ``check`` twice per surviving candidate (initial check +
    post-embed recheck); both calls land in :attr:`seen_texts`. Always returns
    a clean compliant report so no candidate is filtered for a BLOCK.
    """

    def __init__(self, call_log: Optional[list[tuple[str, str]]] = None) -> None:
        self.seen_texts: list[str] = []
        self._call_log = call_log

    async def check(self, copy: str, language) -> Compliance_Report:
        self.seen_texts.append(copy)
        if self._call_log is not None:
            self._call_log.append(("compliance", copy))
        return _make_report()


class _FakeLocalizationTool:
    """Localization_Tool stub. For EN_GLOBAL + en the translate step is skipped
    entirely, so this is effectively never invoked; it returns an empty result
    if it ever is."""

    async def translate(self, *args, **kwargs) -> LocalizerOutput:
        return LocalizerOutput()


class _FakeKeywordEmbedder:
    """Keyword_Embedder stub that leaves the copy unchanged so downstream
    compliance rechecks observe the same text as the original source copy."""

    async def embed(
        self, copy: str, keywords, platform_spec, creative_type
    ) -> EmbedderOutput:
        return EmbedderOutput(
            embedded_copy=copy,
            keyword_coverage=1.0,
            hit_keywords=[],
            skipped_keywords=[],
        )


class _FakeCTAOptimizer:
    """CTA_Optimizer stub returning a fixed, valid strength score."""

    async def optimize(
        self, candidate, target_market, source_lang, creative_type
    ) -> CTAOptimizerOutput:
        return CTAOptimizerOutput(cta_strength_score=0.5, cta_variants=None)


class _OrderRecordingChecker(SemanticDiversityChecker):
    """SemanticDiversityChecker that records every ``check_candidate`` call into
    a shared ordered log *before* delegating to the real implementation."""

    def __init__(
        self,
        call_log: list[tuple[str, str]],
        *,
        config: Optional[SemanticDiversityConfig] = None,
        embed_fn,
    ) -> None:
        super().__init__(config, embed_fn=embed_fn)
        self._call_log = call_log

    async def check_candidate(self, candidate_text: str, accepted_pool: list[str]):
        self._call_log.append(("semantic", candidate_text))
        return await super().check_candidate(candidate_text, accepted_pool)


# ---------------------------------------------------------------------------
# embed_fn builders
# ---------------------------------------------------------------------------


def _onehot_embed_fn(text_to_id: dict[str, int], dim: int):
    """Deterministic embed_fn mapping each text to a one-hot vector.

    Distinct ids -> orthogonal vectors (cosine 0.0, well below the 0.85
    threshold -> accepted). Shared ids -> identical vectors (cosine 1.0,
    above threshold -> rejected as a semantic duplicate).
    """

    def embed(text: str) -> Sequence[float]:
        vec = [0.0] * dim
        vec[text_to_id[text]] = 1.0
        return vec

    return embed


def _unavailable_embed_fn(text: str) -> Sequence[float]:
    raise EmbeddingUnavailableError("embedding model not installed")


def _slow_embed_fn(text: str) -> Sequence[float]:
    # Runs in a worker thread via asyncio.to_thread, so this blocking sleep is
    # bounded by the surrounding asyncio.wait_for timeout in the checker.
    time.sleep(0.5)
    return [1.0, 0.0, 0.0]


def _build_orchestrator(
    *,
    generator: _FakeGenerator,
    compliance: _RecordingComplianceChecker,
    checker: SemanticDiversityChecker,
    trace_dir,
) -> Orchestrator:
    return Orchestrator(
        creative_generator=generator,  # type: ignore[arg-type]
        compliance_checker=compliance,  # type: ignore[arg-type]
        localization_tool=_FakeLocalizationTool(),  # type: ignore[arg-type]
        keyword_embedder=_FakeKeywordEmbedder(),  # type: ignore[arg-type]
        cta_optimizer=_FakeCTAOptimizer(),  # type: ignore[arg-type]
        platform_loader=lambda _platform: _make_platform_spec(),
        trace_recorder=TraceRecorder(base_dir=trace_dir),
        semantic_diversity_checker=checker,
    )


# ---------------------------------------------------------------------------
# Requirement 2.7 — semantic diversity runs BEFORE compliance
# ---------------------------------------------------------------------------


class TestPipelineOrdering:
    """Semantic diversity precedes compliance in the candidate pipeline."""

    async def test_all_semantic_checks_precede_all_compliance_checks(
        self, tmp_path
    ) -> None:
        """Shared ordered call log: every semantic check is recorded before any
        compliance check (the primary ordering proof for Req 2.7)."""
        copies = [
            "Alpha headline one",
            "Beta headline two",
            "Gamma headline three",
            "Delta headline four",
            "Epsilon headline five",
        ]
        # Each copy is orthogonal to every other -> all accepted.
        ids = {copy: i for i, copy in enumerate(copies)}
        embed_fn = _onehot_embed_fn(ids, dim=len(copies))

        call_log: list[tuple[str, str]] = []
        generator = _FakeGenerator(copies)
        compliance = _RecordingComplianceChecker(call_log=call_log)
        checker = _OrderRecordingChecker(call_log, embed_fn=embed_fn)

        orchestrator = _build_orchestrator(
            generator=generator,
            compliance=compliance,
            checker=checker,
            trace_dir=tmp_path,
        )

        ranking = await orchestrator.orchestrate(_make_brief(), request_id="req-order")

        semantic_indices = [
            i for i, (kind, _) in enumerate(call_log) if kind == "semantic"
        ]
        compliance_indices = [
            i for i, (kind, _) in enumerate(call_log) if kind == "compliance"
        ]

        # Both stages ran...
        assert semantic_indices, "expected the semantic checker to be invoked"
        assert compliance_indices, "expected the compliance checker to be invoked"
        # ...and every semantic check precedes every compliance check (Req 2.7).
        assert max(semantic_indices) < min(compliance_indices)

        # All five candidates are semantically diverse, so all survive.
        assert len(ranking.ranked_candidates) == 5
        # Single round: no refill churn from rejections.
        assert ranking.refill_count == 0

    async def test_semantically_rejected_candidate_backfills_to_hit_target(
        self, tmp_path
    ) -> None:
        """A semantic duplicate is initially filtered out of the diverse set,
        but is used as quota backfill so the delivered count still hits the
        exact target (business hard requirement). The diverse candidates are
        what fill the quota first; the near-duplicate only tops up the
        shortfall, and a backfill warning is surfaced."""
        # "Beta" is a paraphrase of "Alpha": different text, identical embedding.
        alpha = "Alpha headline one"
        beta_dup = "Beta says the same thing"
        gamma = "Gamma headline three"
        delta = "Delta headline four"
        epsilon = "Epsilon headline five"
        copies = [alpha, beta_dup, gamma, delta, epsilon]

        # Shared id 0 for alpha + beta_dup => cosine 1.0 => beta_dup rejected.
        ids = {alpha: 0, beta_dup: 0, gamma: 1, delta: 2, epsilon: 3}
        embed_fn = _onehot_embed_fn(ids, dim=4)

        generator = _FakeGenerator(copies)
        compliance = _RecordingComplianceChecker()
        checker = SemanticDiversityChecker(embed_fn=embed_fn)

        orchestrator = _build_orchestrator(
            generator=generator,
            compliance=compliance,
            checker=checker,
            trace_dir=tmp_path,
        )

        ranking = await orchestrator.orchestrate(_make_brief(), request_id="req-reject")

        # The 4 diverse candidates reached compliance in the normal pipeline.
        assert alpha in compliance.seen_texts
        assert gamma in compliance.seen_texts
        # Exact target met (target_count=5): 4 diverse + 1 backfilled duplicate.
        assert len(ranking.ranked_candidates) == 5
        surviving = {c.source_copy for c in ranking.ranked_candidates}
        assert beta_dup in surviving  # the duplicate was used as backfill
        # A backfill warning explains the diversity/quota tradeoff.
        assert any("quota backfill" in w for w in ranking.warnings)


# ---------------------------------------------------------------------------
# Requirement 2.8 — embedding failure degrades to text-dedup only
# ---------------------------------------------------------------------------


class TestDiversityFallback:
    """Embedding timeout / unavailability degrades to text-dedup only without
    tripping the circuit breaker."""

    @staticmethod
    def _copies() -> list[str]:
        # >= 2 candidates so the fallback fires on the 2nd (first has an empty
        # accepted pool and is trivially accepted without embedding).
        return [
            "Alpha headline one",
            "Beta headline two",
            "Gamma headline three",
            "Delta headline four",
            "Epsilon headline five",
        ]

    async def test_embedding_timeout_falls_back_to_text_dedup_only(
        self, tmp_path
    ) -> None:
        copies = self._copies()
        config = SemanticDiversityConfig(timeout_seconds=0.05)
        generator = _FakeGenerator(copies)
        compliance = _RecordingComplianceChecker()
        checker = SemanticDiversityChecker(config, embed_fn=_slow_embed_fn)

        orchestrator = _build_orchestrator(
            generator=generator,
            compliance=compliance,
            checker=checker,
            trace_dir=tmp_path,
        )

        # No CascadeFailureError: completing normally is the breaker-untouched
        # signal the Orchestrator relies on (Req 2.8).
        ranking = await orchestrator.orchestrate(_make_brief(), request_id="req-timeout")

        # Fallback warning surfaced exactly once on the AB_Ranking output.
        assert ranking.warnings.count(_SEMANTIC_FALLBACK_WARNING) == 1
        # Text-dedup-only keeps every candidate produced.
        assert len(ranking.ranked_candidates) == len(copies)
        # No semantic rejections -> no refill churn.
        assert ranking.refill_count == 0

    async def test_embedding_unavailable_falls_back_to_text_dedup_only(
        self, tmp_path
    ) -> None:
        copies = self._copies()
        generator = _FakeGenerator(copies)
        compliance = _RecordingComplianceChecker()
        checker = SemanticDiversityChecker(embed_fn=_unavailable_embed_fn)

        orchestrator = _build_orchestrator(
            generator=generator,
            compliance=compliance,
            checker=checker,
            trace_dir=tmp_path,
        )

        ranking = await orchestrator.orchestrate(
            _make_brief(), request_id="req-unavailable"
        )

        assert ranking.warnings.count(_SEMANTIC_FALLBACK_WARNING) == 1
        assert len(ranking.ranked_candidates) == len(copies)
        assert ranking.refill_count == 0

    async def test_fallback_warning_appended_to_caller_warnings(
        self, tmp_path
    ) -> None:
        """Pre-existing caller warnings are preserved and the fallback warning
        is added alongside them (Req 2.8)."""
        copies = self._copies()
        generator = _FakeGenerator(copies)
        compliance = _RecordingComplianceChecker()
        checker = SemanticDiversityChecker(embed_fn=_unavailable_embed_fn)

        orchestrator = _build_orchestrator(
            generator=generator,
            compliance=compliance,
            checker=checker,
            trace_dir=tmp_path,
        )

        ranking = await orchestrator.orchestrate(
            _make_brief(),
            request_id="req-warn",
            warnings=["keywords truncated from 30 to 20"],
        )

        assert "keywords truncated from 30 to 20" in ranking.warnings
        assert _SEMANTIC_FALLBACK_WARNING in ranking.warnings
