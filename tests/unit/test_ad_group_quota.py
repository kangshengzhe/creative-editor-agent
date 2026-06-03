"""Unit tests for the Ad-Group Quota (Requirement 5).

Verifies that a single creative request drives generation toward the per-type
Target_Count — 15 headlines, 10 descriptions, 5 for CTA/long copy — and that
the AB_Ranking reports that target so consumers can detect under-fill.

Covered acceptance criteria:

* 5.1 — Target_Count derived from creative_type via the Ad_Group_Quota map.
* 5.2 — Orchestrator drives generation toward Target_Count (not the old 5).
* 5.3 — single source of truth shared by Orchestrator and Generator.
* 5.5 — between the viability floor (3) and Target_Count → warn, don't fail.
* 5.6 — below the floor → degraded failure (unchanged).
* 5.7 — AB_Ranking.target_count reports the request's target.
* 5.8 — brief.target_count overrides the default.

The Orchestrator is driven with lightweight in-memory tool fakes (no LLM / ML
dependency). The fake generator produces a configurable number of UNIQUE
compliant candidates per ``generate`` call so the orchestrator's refill loop
can actually reach a 15/10 target across rounds.
"""

from __future__ import annotations

from typing import Optional

import pytest

from creative_agent.config.ad_group_quota import target_count_for
from creative_agent.errors import DegradedFailureError
from creative_agent.models import (
    Compliance_Report,
    Creative_Brief,
    Creative_Candidate,
    Creative_Type,
    Target_Market,
    Target_Platform,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.observability.trace import TraceRecorder
from creative_agent.orchestrator.orchestrator import Orchestrator
from creative_agent.tools.creative_generator import GeneratorOutput
from creative_agent.tools.types import (
    CTAOptimizerOutput,
    EmbedderOutput,
    LocalizerOutput,
)


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------


def _make_report() -> Compliance_Report:
    return Compliance_Report(
        compliance_score=1.0,
        violations=[],
        checked_at="2026-01-01T00:00:00Z",
        checker_version="test",
    )


def _make_candidate(index: int, copy: str, creative_type: Creative_Type) -> Creative_Candidate:
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
        creative_type=creative_type,
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


def _make_brief(
    creative_type: Creative_Type,
    *,
    target_count: Optional[int] = None,
) -> Creative_Brief:
    return Creative_Brief(
        campaign_topic="Top up promo",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=Target_Market.EN_GLOBAL,
        creative_type=creative_type,
        source_language="en",
        keywords=[],
        target_count=target_count,
    )


# ---------------------------------------------------------------------------
# Tool fakes
# ---------------------------------------------------------------------------


class _CountingGenerator:
    """Generator stub that yields ``per_call`` unique candidates each call,
    honouring the ``exclude_copies`` set so the orchestrator's refill loop
    accumulates toward the requested ``min_count``.

    A ``cap`` bounds the total number of distinct candidates the (fake) LLM can
    ever produce, used to exercise the under-fill path (Requirement 5.5).
    """

    supports_angle_generation = False

    def __init__(self, *, creative_type: Creative_Type, per_call: int = 6, cap: int = 1000) -> None:
        self._creative_type = creative_type
        self._per_call = per_call
        self._cap = cap
        self._produced = 0
        self.observed_min_counts: list[int] = []

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
        self.observed_min_counts.append(min_count)
        candidates: list[Creative_Candidate] = []
        for _ in range(self._per_call):
            if self._produced >= self._cap:
                break
            idx = self._produced
            self._produced += 1
            candidates.append(
                _make_candidate(idx, f"copy-{idx}", self._creative_type)
            )
        return GeneratorOutput(candidates=candidates, generation_time_ms=1)


class _CleanCompliance:
    async def check(self, copy: str, language) -> Compliance_Report:
        return _make_report()


class _FakeLocalizationTool:
    async def translate(self, *args, **kwargs) -> LocalizerOutput:
        return LocalizerOutput()


class _FakeKeywordEmbedder:
    async def embed(self, copy, keywords, platform_spec, creative_type) -> EmbedderOutput:
        return EmbedderOutput(
            embedded_copy=copy,
            keyword_coverage=1.0,
            hit_keywords=[],
            skipped_keywords=[],
        )


class _FakeCTAOptimizer:
    async def optimize(self, candidate, target_market, source_lang, creative_type) -> CTAOptimizerOutput:
        return CTAOptimizerOutput(cta_strength_score=0.5, cta_variants=None)


def _build_orchestrator(generator, trace_dir) -> Orchestrator:
    return Orchestrator(
        creative_generator=generator,  # type: ignore[arg-type]
        compliance_checker=_CleanCompliance(),  # type: ignore[arg-type]
        localization_tool=_FakeLocalizationTool(),  # type: ignore[arg-type]
        keyword_embedder=_FakeKeywordEmbedder(),  # type: ignore[arg-type]
        cta_optimizer=_FakeCTAOptimizer(),  # type: ignore[arg-type]
        platform_loader=lambda _platform: _make_platform_spec(),
        trace_recorder=TraceRecorder(base_dir=trace_dir),
    )


# ---------------------------------------------------------------------------
# Requirement 5.1 / 5.3 — quota mapping is the single source of truth
# ---------------------------------------------------------------------------


class TestQuotaMapping:
    def test_target_count_per_type(self) -> None:
        assert target_count_for(Creative_Type.HEADLINE) == 15
        assert target_count_for(Creative_Type.DESCRIPTION) == 10
        assert target_count_for(Creative_Type.CTA) == 5
        assert target_count_for(Creative_Type.LONG_COPY) == 5


# ---------------------------------------------------------------------------
# Requirement 5.2 / 5.7 — generation is driven toward the per-type target
# ---------------------------------------------------------------------------


class TestQuotaDrivesGeneration:
    async def test_headline_targets_15(self, tmp_path) -> None:
        gen = _CountingGenerator(creative_type=Creative_Type.HEADLINE, per_call=6)
        orch = _build_orchestrator(gen, tmp_path)

        ranking = await orch.orchestrate(
            _make_brief(Creative_Type.HEADLINE), request_id="req-hl"
        )

        # 5.2: generation is driven to AT LEAST the quota. The orchestrator
        # overshoots (~1.7x) so semantic/compliance dedup losses still leave a
        # full ad group; the delivered set is truncated back to exactly 15.
        assert gen.observed_min_counts[0] >= 15
        # 5.7: the ranking reports the target.
        assert ranking.target_count == 15
        # Delivered set is truncated to exactly the platform cap of 15, even
        # though the generator can overshoot (6 per call across rounds).
        assert len(ranking.ranked_candidates) == 15
        # No under-fill warning when the target is met.
        assert not any("under-filled" in w for w in ranking.warnings)

    async def test_description_targets_10(self, tmp_path) -> None:
        gen = _CountingGenerator(creative_type=Creative_Type.DESCRIPTION, per_call=4)
        orch = _build_orchestrator(gen, tmp_path)

        ranking = await orch.orchestrate(
            _make_brief(Creative_Type.DESCRIPTION), request_id="req-desc"
        )

        assert gen.observed_min_counts[0] >= 10
        assert ranking.target_count == 10
        assert len(ranking.ranked_candidates) == 10

    async def test_cta_keeps_default_5(self, tmp_path) -> None:
        gen = _CountingGenerator(creative_type=Creative_Type.CTA, per_call=6)
        orch = _build_orchestrator(gen, tmp_path)

        ranking = await orch.orchestrate(
            _make_brief(Creative_Type.CTA), request_id="req-cta"
        )

        assert ranking.target_count == 5
        assert len(ranking.ranked_candidates) == 5

    async def test_overshoot_truncated_to_exact_target(self, tmp_path) -> None:
        """Generator overshoots (per_call=6 → first round yields 18 for a 15
        target); the delivered set is truncated to exactly 15 (platform cap),
        and total_candidates_generated still records the full pre-truncation
        count so the optimise-then-select pool is observable."""
        gen = _CountingGenerator(creative_type=Creative_Type.HEADLINE, per_call=18)
        orch = _build_orchestrator(gen, tmp_path)

        ranking = await orch.orchestrate(
            _make_brief(Creative_Type.HEADLINE), request_id="req-overshoot"
        )

        assert len(ranking.ranked_candidates) == 15
        assert ranking.target_count == 15
        # 18 were generated; only the top 15 are delivered. The surplus is not
        # counted as BLOCK filtering.
        assert ranking.total_candidates_generated == 18
        assert ranking.total_candidates_filtered_out == 0


# ---------------------------------------------------------------------------
# Requirement 5.8 — explicit per-request override
# ---------------------------------------------------------------------------


class TestQuotaOverride:
    async def test_brief_target_count_overrides_default(self, tmp_path) -> None:
        gen = _CountingGenerator(creative_type=Creative_Type.HEADLINE, per_call=5)
        orch = _build_orchestrator(gen, tmp_path)

        ranking = await orch.orchestrate(
            _make_brief(Creative_Type.HEADLINE, target_count=8),
            request_id="req-override",
        )

        assert gen.observed_min_counts[0] >= 8
        assert ranking.target_count == 8
        assert len(ranking.ranked_candidates) == 8


# ---------------------------------------------------------------------------
# Requirement 5.5 — under-fill warns, does not fail
# ---------------------------------------------------------------------------


class TestUnderFill:
    async def test_under_fill_returns_partial_with_warning(self, tmp_path) -> None:
        # cap=6 means the (fake) LLM can never reach the 15 target, but stays
        # above the viability floor of 3 → partial result + warning, no error.
        gen = _CountingGenerator(creative_type=Creative_Type.HEADLINE, per_call=6, cap=6)
        orch = _build_orchestrator(gen, tmp_path)

        ranking = await orch.orchestrate(
            _make_brief(Creative_Type.HEADLINE), request_id="req-underfill"
        )

        assert ranking.target_count == 15
        assert 3 <= len(ranking.ranked_candidates) < 15
        assert any("under-filled" in w for w in ranking.warnings)
        # refill_count must respect the AB_Ranking le=2 cap even though the
        # target was never reached across all rounds.
        assert ranking.refill_count <= 2


# ---------------------------------------------------------------------------
# Requirement 5.6 — below the floor still degrades to failure
# ---------------------------------------------------------------------------


class TestBelowFloorStillFails:
    async def test_below_viability_floor_raises(self, tmp_path) -> None:
        # cap=2 → fewer than the floor of 3 compliant candidates ever produced.
        gen = _CountingGenerator(creative_type=Creative_Type.HEADLINE, per_call=2, cap=2)
        orch = _build_orchestrator(gen, tmp_path)

        with pytest.raises(DegradedFailureError):
            await orch.orchestrate(
                _make_brief(Creative_Type.HEADLINE), request_id="req-floor"
            )
