"""Tests for the fit-first post-processing policy (方案 4).

A copy the LLM writes WITHIN the char limit is a complete sentence and is
always preferred. A copy that OVERFLOWS is only kept (word-boundary truncated)
when a call produced no fitting copy at all — so the delivered ad group is made
of complete sentences in the normal case and a clipped fragment is a rare last
resort, never the default. This directly implements the operator's rule:
"宁可短，也不要被生硬截断".
"""

from __future__ import annotations

from creative_agent.llm.mock_client import MockLLMClient
from creative_agent.models.brief import Creative_Brief
from creative_agent.models.enums import (
    Creative_Type,
    Target_Market,
    Target_Platform,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.tools.creative_generator import CreativeGenerator


def _brief(**overrides) -> Creative_Brief:
    base = dict(
        campaign_topic="Game top-up promo",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=Target_Market.EN_GLOBAL,
        creative_type=Creative_Type.HEADLINE,
        source_language="en",
        keywords=[],
    )
    base.update(overrides)
    return Creative_Brief(**base)


def _spec(headline_limit: int = 30) -> Platform_Spec:
    return Platform_Spec(
        platform=Target_Platform.GOOGLE_ADS,
        char_limits={
            Creative_Type.HEADLINE: headline_limit,
            Creative_Type.DESCRIPTION: 200,
            Creative_Type.CTA: 30,
            Creative_Type.LONG_COPY: 300,
        },
        allowed_creative_types=[Creative_Type.HEADLINE],
    )


def _post(gen: CreativeGenerator, raw: list[str], limit: int = 30):
    return gen._post_process(
        raw_copies=raw,
        brief=_brief(),
        char_limit=limit,
        excludes=[],
        request_id="req-fit",
        generation_language="en",
        platform_spec=_spec(limit),
    )


class TestFitFirst:
    def test_overflow_dropped_when_fitting_copies_exist(self) -> None:
        gen = CreativeGenerator(MockLLMClient())
        fits_a = "Fast topup bonus now"          # 20 chars, fits
        fits_b = "Instant game credits"          # 20 chars, fits
        overflow = "Get 20% more game credits on every single topup today"  # > 30
        out = _post(gen, [fits_a, overflow, fits_b], limit=30)

        texts = [c.source_copy for c in out]
        # Only the two complete, in-limit copies survive.
        assert fits_a in texts
        assert fits_b in texts
        # The overflow copy is NOT delivered (neither full nor truncated),
        # because complete copies were available.
        assert all(len(t) <= 30 for t in texts)
        assert not any(t.startswith("Get 20% more game credits on") for t in texts)

    def test_all_overflow_falls_back_to_word_boundary_truncation(self) -> None:
        gen = CreativeGenerator(MockLLMClient())
        # Every copy overflows -> fallback keeps word-boundary-truncated copy so
        # the call is not empty (generation can still progress).
        raw = [
            "Get 20% more game credits on every single topup today",
            "Reliable instant delivery for every gamer across the world",
        ]
        out = _post(gen, raw, limit=30)
        assert out, "fallback must keep at least one copy when none fit"
        for c in out:
            t = c.source_copy
            assert len(t) <= 30
            # No mid-word cut: result is a prefix ending on a whole word.
            assert not t.endswith(("o", "y")) or t in raw  # heuristic guard
            # No dangling trailing separator.
            assert not t.endswith((",", "-", "—", ":", ";"))

    def test_fitting_copy_preferred_over_truncated_duplicate(self) -> None:
        gen = CreativeGenerator(MockLLMClient())
        # An overflow copy whose truncation would equal a fitting copy must not
        # block the complete one.
        fitting = "Trusted topup bonus today"  # fits
        overflow = "Trusted topup bonus today and so much more value inside"  # > 30
        out = _post(gen, [overflow, fitting], limit=30)
        texts = [c.source_copy for c in out]
        assert fitting in texts
        assert all(len(t) <= 30 for t in texts)

    def test_short_copies_all_kept_unchanged(self) -> None:
        gen = CreativeGenerator(MockLLMClient())
        raw = ["Top up now", "Play and win", "Bonus inside"]
        out = _post(gen, raw, limit=30)
        assert sorted(c.source_copy for c in out) == sorted(raw)
