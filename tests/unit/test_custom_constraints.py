"""Unit tests for operator custom constraints (requirement 6).

Covers the input-side controls that let an operator steer a single request
like prompting an AI:

* ``must_include`` / ``extra_instructions`` — appended to the generation prompt
* ``must_avoid``                            — appended to the prompt AND hard-
  filtered out of the output (LLMs don't always obey, so we enforce)
* ``regenerate_avoid``                      — exclusions for fresh re-runs

The prompt-construction tests assert the rendered prompt text; the must-avoid
enforcement test drives the real ``_post_process`` to prove banned phrases can
never reach the output even if the model emits them.
"""

from __future__ import annotations

from creative_agent.models.brief import Creative_Brief
from creative_agent.models.enums import (
    Creative_Type,
    Target_Market,
    Target_Platform,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.tools.creative_generator import CreativeGenerator
from creative_agent.llm.mock_client import MockLLMClient


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


def _spec() -> Platform_Spec:
    return Platform_Spec(
        platform=Target_Platform.GOOGLE_ADS,
        char_limits={
            Creative_Type.HEADLINE: 90,
            Creative_Type.DESCRIPTION: 200,
            Creative_Type.CTA: 30,
            Creative_Type.LONG_COPY: 300,
        },
        allowed_creative_types=[Creative_Type.HEADLINE],
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptIncludesConstraints:
    def _build(self, brief: Creative_Brief) -> str:
        return CreativeGenerator._build_prompt(
            brief=brief,
            platform_spec=_spec(),
            char_limit=90,
            request_count=5,
            excludes=[],
            output_language="en",
        )

    def test_must_include_in_prompt(self) -> None:
        prompt = self._build(_brief(must_include=["limited offer"]))
        assert "必须包含" in prompt
        assert "limited offer" in prompt

    def test_must_avoid_in_prompt(self) -> None:
        prompt = self._build(_brief(must_avoid=["guaranteed", "jackpot"]))
        assert "必须规避" in prompt
        assert "guaranteed" in prompt
        assert "jackpot" in prompt

    def test_extra_instructions_in_prompt(self) -> None:
        prompt = self._build(_brief(extra_instructions="use a playful tone"))
        assert "额外创作要求" in prompt
        assert "use a playful tone" in prompt

    def test_no_constraints_no_sections(self) -> None:
        prompt = self._build(_brief())
        assert "必须包含" not in prompt
        assert "必须规避" not in prompt
        assert "额外创作要求" not in prompt


# ---------------------------------------------------------------------------
# must_avoid hard enforcement in _post_process
# ---------------------------------------------------------------------------


class TestMustAvoidHardFilter:
    def test_banned_candidates_dropped(self) -> None:
        gen = CreativeGenerator(MockLLMClient())
        brief = _brief(must_avoid=["jackpot"])
        raw = [
            "Top up and win big today",          # clean -> kept
            "Hit the jackpot every weekend",     # banned -> dropped
            "Fast secure game credits",          # clean -> kept
            "Guaranteed JACKPOT for everyone",   # banned (case-insensitive) -> dropped
        ]
        candidates = gen._post_process(
            raw_copies=raw,
            brief=brief,
            char_limit=90,
            excludes=[],
            request_id="t",
            generation_language="en",
            platform_spec=_spec(),
        )
        copies = [c.source_copy for c in candidates]
        assert "Top up and win big today" in copies
        assert "Fast secure game credits" in copies
        # No surviving candidate contains the banned word, in any case.
        assert all("jackpot" not in c.lower() for c in copies)
        assert len(candidates) == 2

    def test_no_must_avoid_keeps_all(self) -> None:
        gen = CreativeGenerator(MockLLMClient())
        brief = _brief()  # no must_avoid
        raw = ["Hit the jackpot", "Top up now"]
        candidates = gen._post_process(
            raw_copies=raw,
            brief=brief,
            char_limit=90,
            excludes=[],
            request_id="t",
            generation_language="en",
            platform_spec=_spec(),
        )
        assert len(candidates) == 2


# ---------------------------------------------------------------------------
# regenerate_avoid feeds the dedup exclude set
# ---------------------------------------------------------------------------


class TestRegenerateAvoid:
    def test_previous_copies_excluded(self) -> None:
        gen = CreativeGenerator(MockLLMClient())
        brief = _brief()
        previous = "Top up now and play"
        # Simulate the orchestrator passing regenerate_avoid via excludes.
        candidates = gen._post_process(
            raw_copies=[previous, "A brand new different headline"],
            brief=brief,
            char_limit=90,
            excludes=[previous],
            request_id="t",
            generation_language="en",
            platform_spec=_spec(),
        )
        copies = [c.source_copy for c in candidates]
        assert previous not in copies
        assert "A brand new different headline" in copies
