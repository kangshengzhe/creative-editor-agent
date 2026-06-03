"""Example-based unit tests for CJK display-width limit enforcement.

Feature: creative-localization-diversity (task 11.2).

These tests cover the display-width enforcement wired into
:class:`~creative_agent.tools.creative_generator.CreativeGenerator` (task 11.1):

* For CJK markets (JP, KR, HK, TW) the generator enforces the
  Platform_Spec character limit in **Display_Units** (CJK char = 2 units)
  via the :class:`~creative_agent.integration.display_width.DisplayWidthCalculator`
  rather than ``len()`` (Requirement 4.8), truncating whole characters without
  splitting (Requirement 4.9).
* Every produced candidate carries a ``display_width`` equal to the
  Display_Unit width of its (possibly truncated) ``source_copy`` (Requirement
  4.8).
* Non-CJK English flows keep ``len()`` semantics — for pure-ASCII copy the
  Display_Unit width equals the string length — while ``display_width`` is
  still populated (Requirement 4.13 via 4.8).
* The Platform_Spec ``use_display_width`` flag forces Display_Unit enforcement
  even for a non-CJK market (Requirement 4.10).

Test level
----------

Tests 1–3 drive the public :meth:`CreativeGenerator.generate` entry point — the
highest level that still keeps the behaviour deterministic. A CJK market (JP)
triggers native-language generation, but the injected
:class:`~creative_agent.llm.mock_client.MockLLMClient` resolves *every* prompt
(regardless of the native Japanese system prompt) to a canned batch of CJK
candidate copies via :meth:`set_default_response`. Native generation therefore
"succeeds" with our canned copies and the English-translate fallback (Req 1.6)
is never reached. The canned copies are unique and numerous enough to clear
``min_count`` after truncation + dedup.

Test 4 unit-tests the ``_should_use_display_width`` helper directly to isolate
the Platform_Spec ``use_display_width`` flag from the CJK-market and
wide-character triggers (Requirement 4.10) — the cleanest way to show the flag
alone forces Display_Unit semantics for an otherwise non-CJK, ASCII flow.

The project runs with ``asyncio_mode = "auto"`` (pytest-asyncio), so plain
``async def test_...`` coroutines are awaited automatically.
"""

from __future__ import annotations

from creative_agent.integration.display_width import DisplayWidthCalculator
from creative_agent.llm.mock_client import MockLLMClient
from creative_agent.models.brief import Creative_Brief
from creative_agent.models.enums import (
    Creative_Type,
    Target_Market,
    Target_Platform,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.tools.creative_generator import (
    CreativeGenerator,
    _should_use_display_width,
)

# Independent calculator used by the assertions (the generator uses its own
# shared instance internally; this mirrors it for verification).
_WIDTH = DisplayWidthCalculator()

# HEADLINE limit expressed in Display_Units. Deliberately small so that CJK
# copies whose len() is *under* the limit still exceed it in Display_Units,
# making a len()-based truncation behave differently from a Display_Unit one.
_HEADLINE_WIDTH_LIMIT = 10

# Seven Japanese (CJK) candidate copies sharing a common suffix but each led by
# a *distinct* kanji. Every character is width-2, so each copy's Display_Unit
# width is 2 * len(copy). The distinct leading kanji guarantees the truncated
# prefixes stay unique (so dedup keeps all of them).
_CJK_LEADS = ["春", "夏", "秋", "冬", "海", "山", "空"]
_CJK_TAIL = "限定大特価セール"  # common suffix (all width-2 chars)
_CJK_CANDIDATES = [lead + _CJK_TAIL for lead in _CJK_LEADS]

# Seven distinct, short ASCII English copies for the non-CJK contrast test.
_ASCII_CANDIDATES = [
    "Recharge your game wallet in seconds",
    "Top up instantly and jump back in",
    "Fast, secure game credits anytime",
    "Never run out of coins mid-match",
    "Reload now and keep the streak alive",
    "Quick top-ups for serious gamers",
    "Power up your account in one tap",
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_brief(market: Target_Market) -> Creative_Brief:
    return Creative_Brief(
        campaign_topic="Game top-up promotion",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=market,
        creative_type=Creative_Type.HEADLINE,
        selling_points=["Fast top-ups", "Secure payments"],
        target_audience="Mobile gamers",
        source_language="en",
        brand_name="Coco",
    )


def _make_platform_spec(
    *,
    headline_limit: int,
    use_display_width: bool = False,
) -> Platform_Spec:
    return Platform_Spec(
        platform=Target_Platform.GOOGLE_ADS,
        char_limits={
            Creative_Type.HEADLINE: headline_limit,
            Creative_Type.DESCRIPTION: 200,
            Creative_Type.CTA: 30,
            Creative_Type.LONG_COPY: 300,
        },
        allowed_creative_types=[
            Creative_Type.HEADLINE,
            Creative_Type.DESCRIPTION,
            Creative_Type.CTA,
            Creative_Type.LONG_COPY,
        ],
        use_display_width=use_display_width,
    )


def _make_generator(llm: MockLLMClient) -> CreativeGenerator:
    return CreativeGenerator(llm, timeout_ms=10_000)


def _make_mock_llm(copies: list[str]) -> MockLLMClient:
    """A mock LLM whose default response returns ``copies`` for any prompt.

    ``set_default_response`` ignores the system prompt, so it is returned even
    for the native Japanese generation prompt — letting native generation
    "succeed" with our canned CJK copies (no English-translate fallback).
    """
    llm = MockLLMClient()
    llm.set_default_response({"candidates": [{"copy": c} for c in copies]})
    return llm


# ---------------------------------------------------------------------------
# Test 1 — CJK truncation enforced by Display_Units, not len() (Req 4.8 / 4.9)
# ---------------------------------------------------------------------------


class TestCJKTruncationByDisplayUnits:
    """CJK candidates are validated/truncated by Display_Units, not len()."""

    def test_premise_naive_len_would_not_truncate(self) -> None:
        """Sanity: each canned CJK copy exceeds the limit in Display_Units but
        NOT in raw character count — so a len()-based truncation would leave it
        untouched while a Display_Unit truncation must shorten it."""
        for copy in _CJK_CANDIDATES:
            # len() is under the limit -> a naive len()-based check passes.
            assert len(copy) <= _HEADLINE_WIDTH_LIMIT
            # Display_Unit width is over the limit -> Display_Unit check fails.
            assert _WIDTH.text_width(copy) > _HEADLINE_WIDTH_LIMIT

    async def test_returned_candidates_fit_display_unit_limit(self) -> None:
        llm = _make_mock_llm(_CJK_CANDIDATES)
        generator = _make_generator(llm)
        spec = _make_platform_spec(headline_limit=_HEADLINE_WIDTH_LIMIT)

        output = await generator.generate(
            _make_brief(Target_Market.JP),
            spec,
            min_count=5,
            request_id="req-cjk-width",
        )

        assert len(output.candidates) >= 5
        for candidate in output.candidates:
            # Validated by Display_Units (each CJK char = 2), NOT by len():
            # a 5-char CJK prefix is width 10 (== limit) but len 5.
            assert _WIDTH.text_width(candidate.source_copy) <= _HEADLINE_WIDTH_LIMIT

    async def test_truncation_removes_whole_characters(self) -> None:
        """Truncation keeps a character-boundary prefix of an original copy —
        no multi-byte character is split (Req 4.9)."""
        llm = _make_mock_llm(_CJK_CANDIDATES)
        generator = _make_generator(llm)
        spec = _make_platform_spec(headline_limit=_HEADLINE_WIDTH_LIMIT)

        output = await generator.generate(
            _make_brief(Target_Market.JP),
            spec,
            min_count=5,
            request_id="req-cjk-width",
        )

        for candidate in output.candidates:
            copy = candidate.source_copy
            # The result is a prefix of exactly one original (untruncated) copy:
            # proof that truncation only dropped trailing whole characters.
            matching = [o for o in _CJK_CANDIDATES if o.startswith(copy)]
            assert matching, f"{copy!r} is not a prefix of any original copy"
            # And it is strictly shorter than that original (it WAS truncated),
            # confirming Display_Unit enforcement actually fired.
            assert all(len(copy) < len(o) for o in matching)
            # Every retained character is itself a whole width-2 CJK char.
            assert all(_WIDTH.char_width(ch) == 2 for ch in copy)

    async def test_distinct_candidates_survive_dedup(self) -> None:
        """The distinct leading kanji keep truncated prefixes unique, so dedup
        does not collapse the batch below ``min_count``."""
        llm = _make_mock_llm(_CJK_CANDIDATES)
        generator = _make_generator(llm)
        spec = _make_platform_spec(headline_limit=_HEADLINE_WIDTH_LIMIT)

        output = await generator.generate(
            _make_brief(Target_Market.JP),
            spec,
            min_count=5,
            request_id="req-cjk-width",
        )

        copies = [c.source_copy for c in output.candidates]
        assert len(copies) == len(set(copies))


# ---------------------------------------------------------------------------
# Test 2 — display_width populated correctly for CJK copies (Req 4.8)
# ---------------------------------------------------------------------------


class TestDisplayWidthPopulatedForCJK:
    """Each candidate's ``display_width`` equals the Display_Unit width."""

    async def test_display_width_matches_calculator(self) -> None:
        llm = _make_mock_llm(_CJK_CANDIDATES)
        generator = _make_generator(llm)
        spec = _make_platform_spec(headline_limit=_HEADLINE_WIDTH_LIMIT)

        output = await generator.generate(
            _make_brief(Target_Market.JP),
            spec,
            min_count=5,
            request_id="req-cjk-width",
        )

        for candidate in output.candidates:
            expected = _WIDTH.text_width(candidate.source_copy)
            assert candidate.display_width == expected

    async def test_display_width_equals_expected_unit_count(self) -> None:
        """A truncated CJK copy that fills the limit has display_width == limit
        and is exactly half as many characters (each char = 2 units)."""
        llm = _make_mock_llm(_CJK_CANDIDATES)
        generator = _make_generator(llm)
        spec = _make_platform_spec(headline_limit=_HEADLINE_WIDTH_LIMIT)

        output = await generator.generate(
            _make_brief(Target_Market.JP),
            spec,
            min_count=5,
            request_id="req-cjk-width",
        )

        for candidate in output.candidates:
            # All copies were truncated to fill the limit exactly (10 units = 5
            # CJK chars), since each original is longer than 5 chars.
            assert candidate.display_width == _HEADLINE_WIDTH_LIMIT
            assert candidate.display_width == 2 * len(candidate.source_copy)


# ---------------------------------------------------------------------------
# Test 3 — non-CJK ASCII flow keeps len() semantics (Req 4.13 via 4.8)
# ---------------------------------------------------------------------------


class TestASCIIFlowPreservesLenSemantics:
    """For a non-CJK English market with ASCII copy, width == len()."""

    async def test_display_width_equals_string_length(self) -> None:
        llm = _make_mock_llm(_ASCII_CANDIDATES)
        generator = _make_generator(llm)
        # Generous limit so the short ASCII copies are not truncated.
        spec = _make_platform_spec(headline_limit=90)

        output = await generator.generate(
            _make_brief(Target_Market.EN_GLOBAL),
            spec,
            min_count=5,
            request_id="req-ascii-width",
        )

        assert len(output.candidates) >= 5
        for candidate in output.candidates:
            copy = candidate.source_copy
            # ASCII-only -> Display_Unit width equals raw character length.
            assert candidate.display_width == len(copy)
            assert _WIDTH.text_width(copy) == len(copy)


# ---------------------------------------------------------------------------
# Test 4 — use_display_width flag forces Display_Unit enforcement (Req 4.10)
# ---------------------------------------------------------------------------


class TestUseDisplayWidthFlagForcesDisplayUnits:
    """The Platform_Spec ``use_display_width`` flag is an independent trigger."""

    def test_flag_forces_display_width_for_non_cjk_ascii_flow(self) -> None:
        """A non-CJK market with ASCII text uses len() by default, but the
        ``use_display_width`` flag forces Display_Unit semantics (Req 4.10)."""
        brief = _make_brief(Target_Market.EN_GLOBAL)
        ascii_text = "Top up now"

        spec_len = _make_platform_spec(headline_limit=30, use_display_width=False)
        spec_units = _make_platform_spec(headline_limit=30, use_display_width=True)

        # Default: non-CJK market + ASCII text -> len() semantics.
        assert _should_use_display_width(brief, spec_len, ascii_text) is False
        # Flag set: Display_Unit semantics forced regardless of market/text.
        assert _should_use_display_width(brief, spec_units, ascii_text) is True

    def test_cjk_market_and_wide_char_are_independent_triggers(self) -> None:
        """A CJK market, or copy containing a width-2 char, each independently
        select Display_Unit semantics even without the flag (Req 4.8)."""
        spec = _make_platform_spec(headline_limit=30, use_display_width=False)

        # CJK market triggers Display_Unit semantics on its own.
        assert (
            _should_use_display_width(_make_brief(Target_Market.JP), spec, "ascii")
            is True
        )
        # Non-CJK market + wide character in the copy also triggers it.
        assert (
            _should_use_display_width(
                _make_brief(Target_Market.EN_GLOBAL), spec, "限定セール"
            )
            is True
        )
        # Non-CJK market + pure ASCII -> stays on len() semantics.
        assert (
            _should_use_display_width(
                _make_brief(Target_Market.EN_GLOBAL), spec, "limited sale"
            )
            is False
        )
