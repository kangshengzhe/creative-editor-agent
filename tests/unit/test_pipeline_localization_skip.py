"""Tests for the pipeline localization step's redundant-translation skip.

Covers task 8.2 / Requirement 1.4: when ``brief.source_language`` equals the
target market's primary language, the Localization_Tool translate step for
that language is skipped. When no other market languages remain, the translate
call is bypassed entirely. All other source/primary combinations preserve the
pre-1.4 behaviour (``target_languages=None`` ⇒ derive the full list).
"""

from __future__ import annotations

import pytest

from creative_agent.models import (
    Compliance_Report,
    Creative_Brief,
    Creative_Candidate,
    Creative_Type,
    Target_Language,
    Target_Market,
    Target_Platform,
)
from creative_agent.orchestrator.pipeline import (
    PipelineDeps,
    _resolve_target_languages,
    _resolve_source_language,
    _run_localization,
)
from creative_agent.tools.types import LocalizerOutput


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class _RecordingLocalizationTool:
    """Fake Localization_Tool that records the kwargs of each translate call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def translate(
        self,
        source_copy,
        source_language=Target_Language.EN,
        target_languages=None,
        target_market=Target_Market.EN_GLOBAL,
    ) -> LocalizerOutput:
        self.calls.append(
            {
                "source_copy": source_copy,
                "source_language": source_language,
                "target_languages": target_languages,
                "target_market": target_market,
            }
        )
        out = LocalizerOutput()
        # Echo back a translation for every requested language so the result
        # is well-formed; the test only asserts on the recorded call args.
        langs = (
            target_languages
            if target_languages is not None
            else []
        )
        for lang in langs:
            out.localized_versions[lang] = f"{source_copy}::{lang.value}"
        return out


def _make_report() -> Compliance_Report:
    return Compliance_Report(
        compliance_score=1.0,
        violations=[],
        checked_at="2026-01-01T00:00:00Z",
        checker_version="test",
    )


def _make_candidate() -> Creative_Candidate:
    return Creative_Candidate(
        candidate_id="c0",
        generation_index=0,
        source_copy="Top up now and get a bonus",
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


def _make_brief(market: Target_Market, source_language: str) -> Creative_Brief:
    return Creative_Brief(
        campaign_topic="Wallet top-up promo",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=market,
        creative_type=Creative_Type.HEADLINE,
        source_language=source_language,
    )


def _deps(tool: _RecordingLocalizationTool) -> PipelineDeps:
    # Only ``localization_tool`` is exercised by ``_run_localization``; the
    # other deps are irrelevant for this step.
    return PipelineDeps(
        compliance_checker=None,  # type: ignore[arg-type]
        localization_tool=tool,  # type: ignore[arg-type]
        keyword_embedder=None,  # type: ignore[arg-type]
        cta_optimizer=None,  # type: ignore[arg-type]
        request_id="req-1",
        tool_failure_counter=[0],
    )


# ---------------------------------------------------------------------------
# _resolve_target_languages — pure helper
# ---------------------------------------------------------------------------


class TestResolveTargetLanguages:
    def test_source_not_matching_primary_returns_none(self):
        # PH primary is FIL; English source does not match → full derive (None).
        brief = _make_brief(Target_Market.PH, "en")
        langs, skipped = _resolve_target_languages(brief=brief)
        assert langs is None
        assert skipped is None

    def test_source_matches_primary_removes_that_language(self):
        # RU market = [RU, EN]; source "ru" matches primary → translate only EN.
        brief = _make_brief(Target_Market.RU, "ru")
        langs, skipped = _resolve_target_languages(brief=brief)
        assert langs == [Target_Language.EN]
        assert skipped == "ru"

    def test_source_matches_non_english_primary_ph(self):
        # PH market = [FIL, EN]; source "fil" matches primary → translate only EN.
        brief = _make_brief(Target_Market.PH, "fil")
        langs, skipped = _resolve_target_languages(brief=brief)
        assert langs == [Target_Language.EN]
        assert skipped == "fil"

    def test_english_only_market_with_english_source_empties_list(self):
        # US market = [EN]; source "en" matches the only language → nothing left.
        brief = _make_brief(Target_Market.US, "en")
        langs, skipped = _resolve_target_languages(brief=brief)
        assert langs == []
        assert skipped == "en"

    def test_case_insensitive_match(self):
        brief = _make_brief(Target_Market.RU, "RU")
        langs, skipped = _resolve_target_languages(brief=brief)
        assert langs == [Target_Language.EN]
        assert skipped == "ru"


# ---------------------------------------------------------------------------
# _run_localization — integration with the fake tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunLocalizationSkip:
    async def _run(self, market: Target_Market, source_language: str):
        tool = _RecordingLocalizationTool()
        candidate = _make_candidate()
        brief = _make_brief(market, source_language)
        await _run_localization(
            candidate=candidate,
            brief=brief,
            deps=_deps(tool),
            candidate_id=candidate.candidate_id,
            source_lang=_resolve_source_language(brief.source_language),
        )
        return tool, candidate

    async def test_no_match_preserves_full_derive(self):
        # PH + en: source != primary (fil) → translate called with None.
        tool, candidate = await self._run(Target_Market.PH, "en")
        assert len(tool.calls) == 1
        assert tool.calls[0]["target_languages"] is None
        assert tool.calls[0]["target_market"] is Target_Market.PH

    async def test_match_excludes_primary_language(self):
        # RU + ru: skip RU, translate only EN.
        tool, candidate = await self._run(Target_Market.RU, "ru")
        assert len(tool.calls) == 1
        assert tool.calls[0]["target_languages"] == [Target_Language.EN]
        # RU must not have been translated.
        assert Target_Language.RU not in candidate.localized_versions
        assert Target_Language.EN in candidate.localized_versions

    async def test_english_only_market_skips_translate_entirely(self):
        # US + en: no remaining languages → translate not called at all.
        tool, candidate = await self._run(Target_Market.US, "en")
        assert tool.calls == []
        assert candidate.localized_versions == {}
        assert candidate.failed_languages == []

    async def test_en_global_market_skips_translate_entirely(self):
        tool, candidate = await self._run(Target_Market.EN_GLOBAL, "en")
        assert tool.calls == []
        assert candidate.localized_versions == {}

    async def test_ph_fil_source_translates_only_english(self):
        tool, candidate = await self._run(Target_Market.PH, "fil")
        assert len(tool.calls) == 1
        assert tool.calls[0]["target_languages"] == [Target_Language.EN]
