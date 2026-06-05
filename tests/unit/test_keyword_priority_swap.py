"""Unit tests for the keyword-coverage priority swap (Stage B.6).

Business rule (confirmed 2026-06): keyword coverage outranks diversity. If a
delivered candidate is missing an SEO keyword, swap it for a reserve candidate
(a near-duplicate set aside by the semantic filter) that DOES cover every
keyword — even if that replacement is semantically close to existing copy.

These tests drive ``Orchestrator._swap_in_keyword_covering_reserve`` directly
with constructed candidates and minimal tool fakes, so the swap logic is
verified in isolation.
"""

from __future__ import annotations

import pytest

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
from creative_agent.orchestrator.pipeline import PipelineDeps
from creative_agent.tools.types import CTAOptimizerOutput, EmbedderOutput, LocalizerOutput


def _report() -> Compliance_Report:
    return Compliance_Report(
        compliance_score=1.0,
        violations=[],
        checked_at="2026-01-01T00:00:00Z",
        checker_version="test",
    )


def _cand(idx: int, copy: str, *, skipped: list[str] | None = None) -> Creative_Candidate:
    return Creative_Candidate(
        candidate_id=f"c{idx}",
        generation_index=idx,
        source_copy=copy,
        source_language="en",
        compliance_report=_report(),
        keyword_coverage=1.0 if not skipped else 0.5,
        hit_keywords=[],
        skipped_keywords=list(skipped or []),
        cta_strength_score=0.0,
        cta_variants=None,
        localized_versions={},
        failed_languages=[],
        composite_score=0.0,
        generation_language="en",
        target_platform=Target_Platform.GOOGLE_ADS,
        creative_type=Creative_Type.HEADLINE,
        warnings=[],
    )


def _spec() -> Platform_Spec:
    return Platform_Spec(
        platform=Target_Platform.GOOGLE_ADS,
        char_limits={
            Creative_Type.HEADLINE: 30,
            Creative_Type.DESCRIPTION: 90,
            Creative_Type.CTA: 15,
            Creative_Type.LONG_COPY: 200,
        },
        allowed_creative_types=[Creative_Type.HEADLINE],
    )


def _brief() -> Creative_Brief:
    return Creative_Brief(
        campaign_topic="Top up promo",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=Target_Market.EN_GLOBAL,
        creative_type=Creative_Type.HEADLINE,
        source_language="en",
        keywords=["topup", "bonus"],
    )


class _PassThroughCompliance:
    async def check(self, copy: str, language) -> Compliance_Report:
        return _report()


class _FakeLocalization:
    async def translate(self, *a, **k) -> LocalizerOutput:
        return LocalizerOutput()


class _RealisticEmbedder:
    """Reports skipped_keywords based on actual presence in the copy, so a
    reserve candidate that truly contains the keywords comes out complete."""

    async def embed(self, copy, keywords, platform_spec, creative_type, language=None) -> EmbedderOutput:
        low = copy.lower()
        hit = [k for k in keywords if k.lower() in low]
        skipped = [k for k in keywords if k.lower() not in low]
        return EmbedderOutput(
            embedded_copy=copy,
            keyword_coverage=len(hit) / len(keywords) if keywords else 1.0,
            hit_keywords=hit,
            skipped_keywords=skipped,
        )


class _FakeCTA:
    async def optimize(self, *a, **k) -> CTAOptimizerOutput:
        return CTAOptimizerOutput(cta_strength_score=0.5, cta_variants=None)


def _orchestrator(tmp_path) -> Orchestrator:
    return Orchestrator(
        creative_generator=object(),  # type: ignore[arg-type]  — unused here
        compliance_checker=_PassThroughCompliance(),  # type: ignore[arg-type]
        localization_tool=_FakeLocalization(),  # type: ignore[arg-type]
        keyword_embedder=_RealisticEmbedder(),  # type: ignore[arg-type]
        cta_optimizer=_FakeCTA(),  # type: ignore[arg-type]
        platform_loader=lambda _p: _spec(),
        trace_recorder=TraceRecorder(base_dir=tmp_path),
    )


def _deps(orch: Orchestrator) -> PipelineDeps:
    return PipelineDeps(
        compliance_checker=orch._compliance_checker,
        localization_tool=orch._localization_tool,
        keyword_embedder=orch._keyword_embedder,
        cta_optimizer=orch._cta_optimizer,
        request_id="req-swap",
        tool_failure_counter=[0],
    )


class TestKeywordPrioritySwap:
    async def test_missing_keyword_candidate_is_swapped(self, tmp_path) -> None:
        orch = _orchestrator(tmp_path)
        # Delivered: one full-coverage, one missing 'bonus'.
        delivered = [
            _cand(0, "topup bonus now"),
            _cand(1, "great topup deal", skipped=["bonus"]),
        ]
        # Reserve has a near-duplicate that DOES contain both keywords.
        reserve = [_cand(99, "instant topup bonus offer")]
        warnings: list[str] = []

        await orch._swap_in_keyword_covering_reserve(
            delivered=delivered,
            reserve=reserve,
            brief=_brief(),
            platform_spec=_spec(),
            deps=_deps(orch),
            request_warnings=warnings,
            request_id="req-swap",
        )

        copies = [c.source_copy for c in delivered]
        # The keyword-missing copy is gone; the covering reserve copy is in.
        assert "great topup deal" not in copies
        assert "instant topup bonus offer" in copies
        # Count is unchanged.
        assert len(delivered) == 2
        assert any("keyword-priority swap" in w for w in warnings)

    async def test_no_swap_when_all_covered(self, tmp_path) -> None:
        orch = _orchestrator(tmp_path)
        delivered = [_cand(0, "topup bonus now"), _cand(1, "topup and bonus")]
        reserve = [_cand(99, "another topup bonus")]
        warnings: list[str] = []

        await orch._swap_in_keyword_covering_reserve(
            delivered=delivered,
            reserve=reserve,
            brief=_brief(),
            platform_spec=_spec(),
            deps=_deps(orch),
            request_warnings=warnings,
            request_id="req-swap",
        )
        assert [c.source_copy for c in delivered] == ["topup bonus now", "topup and bonus"]
        assert warnings == []

    async def test_no_swap_when_reserve_also_lacks_keyword(self, tmp_path) -> None:
        orch = _orchestrator(tmp_path)
        delivered = [_cand(0, "topup bonus now"), _cand(1, "great topup deal", skipped=["bonus"])]
        # Reserve copies also miss 'bonus' → nothing to swap in.
        reserve = [_cand(99, "cheap topup here"), _cand(98, "topup fast")]
        warnings: list[str] = []

        await orch._swap_in_keyword_covering_reserve(
            delivered=delivered,
            reserve=reserve,
            brief=_brief(),
            platform_spec=_spec(),
            deps=_deps(orch),
            request_warnings=warnings,
            request_id="req-swap",
        )
        # Unchanged — best effort, never worse.
        assert "great topup deal" in [c.source_copy for c in delivered]
        assert warnings == []

    async def test_picks_a_covering_reserve_among_several(self, tmp_path) -> None:
        orch = _orchestrator(tmp_path)
        delivered = [_cand(0, "topup bonus now"), _cand(1, "great topup deal", skipped=["bonus"])]
        # Several covering reserve options; any one of them is an acceptable swap.
        reserve = [_cand(99, "topup bonus combo"), _cand(98, "topup bonus extra")]
        warnings: list[str] = []

        await orch._swap_in_keyword_covering_reserve(
            delivered=delivered,
            reserve=reserve,
            brief=_brief(),
            platform_spec=_spec(),
            deps=_deps(orch),
            request_warnings=warnings,
            request_id="req-swap",
        )
        copies = [c.source_copy for c in delivered]
        assert "great topup deal" not in copies
        assert any(c in copies for c in ["topup bonus combo", "topup bonus extra"])
