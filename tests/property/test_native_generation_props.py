"""Property-based tests for native-generation translation skipping (task 8.3).

Feature: creative-localization-diversity.

Exercises Property 3 from design.md against the orchestrator's per-candidate
localization step
(:func:`creative_agent.orchestrator.pipeline._run_localization`): when a
Creative_Brief's ``source_language`` already matches the target market's
primary language, the copy is native in that language and the
Localization_Tool translate step must NOT be invoked for that language.

The Localization_Tool is replaced with a dependency-free recording fake (mirror
of the one in ``tests/unit/test_pipeline_localization_skip.py``) so the property
observes exactly which ``target_languages`` each translate call requested,
without touching any real LLM. The market's primary language is derived from
:class:`LanguagePromptSelector`, which reads the same ``_MARKET_LANGUAGES``
source of truth the pipeline uses.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from creative_agent.integration.language_prompts import LanguagePromptSelector
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
    _resolve_source_language,
    _run_localization,
)
from creative_agent.tools.types import LocalizerOutput

pytestmark = pytest.mark.property

_SELECTOR = LanguagePromptSelector()


# ---------------------------------------------------------------------------
# Dependency-free recording fake (mirrors test_pipeline_localization_skip.py)
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
        langs = target_languages if target_languages is not None else []
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
    return PipelineDeps(
        compliance_checker=None,  # type: ignore[arg-type]
        localization_tool=tool,  # type: ignore[arg-type]
        keyword_embedder=None,  # type: ignore[arg-type]
        cta_optimizer=None,  # type: ignore[arg-type]
        request_id="req-prop-3",
        tool_failure_counter=[0],
    )


# Feature: creative-localization-diversity, Property 3: Source-target language match skips translation
@settings(max_examples=100)
@given(market=st.sampled_from(list(Target_Market)))
def test_source_matches_primary_skips_translation_for_that_language(
    market: Target_Market,
) -> None:
    """Property 3: Source-target language match skips translation.

    For any Target_Market, when the brief's ``source_language`` equals that
    market's primary language, the Localization_Tool translate step is never
    invoked for the primary language. Concretely: either ``translate`` is not
    called at all (no other market languages remain) OR every recorded call
    requests an explicit ``target_languages`` list that EXCLUDES the primary
    language. The redundant re-translation of the already-native source copy is
    therefore skipped.

    **Validates: Requirements 1.4**
    """
    # The market's primary language is the source language for this brief.
    primary_code = _SELECTOR.get_primary_language(market)
    primary_language = Target_Language(primary_code)

    tool = _RecordingLocalizationTool()
    candidate = _make_candidate()
    brief = _make_brief(market, primary_code)

    asyncio.run(
        _run_localization(
            candidate=candidate,
            brief=brief,
            deps=_deps(tool),
            candidate_id=candidate.candidate_id,
            source_lang=_resolve_source_language(brief.source_language),
        )
    )

    # The translate step must never be asked to translate into the primary
    # language. Each recorded call must carry an explicit list (never None,
    # which would let the tool derive the full market list including the
    # primary) and that list must exclude the primary language.
    for call in tool.calls:
        requested = call["target_languages"]
        assert requested is not None, (
            f"market={market.value}: translate was called with target_languages"
            " derived from the market (None), which would re-translate the "
            f"native primary language {primary_code!r}"
        )
        assert primary_language not in requested, (
            f"market={market.value}: primary language {primary_code!r} must not"
            f" be among the requested translation targets {requested!r}"
        )

    # And the candidate must never end up with a translation for the primary
    # language — proving the redundant translation was skipped end-to-end.
    assert primary_language not in candidate.localized_versions, (
        f"market={market.value}: primary language {primary_code!r} should not"
        " appear in localized_versions when source == primary"
    )
