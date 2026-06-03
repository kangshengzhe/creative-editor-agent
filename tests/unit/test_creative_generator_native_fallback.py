"""Example-based unit tests for the native-generation fallback path.

Feature: creative-localization-diversity (task 8.4).

These tests cover the native-generation fallback semantics described in
Requirement 1.6: when native-language generation fails for a non-English
market after 3 attempts (1 initial + 2 retries), the
:class:`~creative_agent.tools.creative_generator.CreativeGenerator` falls back
to English generation followed by Localization_Tool translation, stamping each
translated candidate with the target ``generation_language``, and logs the
``creative_generator.native_generation_fallback`` event with the target
language, failure reason, and attempt count.

The two collaborators are injected so the tests are fully deterministic and
require no real LLM / ML dependency:

* A stub :class:`LLMClient` whose ``complete_json`` behaves *conditionally on
  the ``system`` argument*: native-language generation (any system prompt that
  is not the standard English ``_SYSTEM_PROMPT``) returns an empty candidate
  list, forcing every native attempt to be a soft failure; the English
  fallback generation (``system == _SYSTEM_PROMPT``) returns a healthy batch of
  candidates so the fallback can succeed.
* A fake Localization_Tool whose async ``translate`` deterministically succeeds
  and records every call, so the test can assert the English-then-translate
  path was actually exercised.

The project runs with ``asyncio_mode = "auto"`` (pytest-asyncio), so plain
``async def test_...`` coroutines are awaited automatically. Structured logs
are captured with ``structlog.testing.capture_logs`` (matching the pattern in
``tests/unit/test_semantic_diversity.py``).
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from structlog.testing import capture_logs

from creative_agent.integration.language_prompts import LanguagePromptSelector
from creative_agent.llm.client import LLMClient
from creative_agent.models import Target_Language
from creative_agent.models.brief import Creative_Brief
from creative_agent.models.enums import (
    Creative_Type,
    Target_Market,
    Target_Platform,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.tools.creative_generator import (
    _MAX_ATTEMPTS,
    _SYSTEM_PROMPT,
    CreativeGenerator,
)
from creative_agent.tools.types import LocalizerOutput

# A non-English market whose primary language is Thai, so native generation is
# active (PH→fil and TH→th both qualify; we use TH here).
_MARKET = Target_Market.TH
_PRIMARY_LANGUAGE = "th"

# English candidates the stub returns for the fallback generation step. Seven
# distinct, short copies so they survive dedup/truncation and clear min_count.
_ENGLISH_CANDIDATES = [
    "Recharge your game wallet in seconds",
    "Top up instantly and jump back in",
    "Fast, secure game credits anytime",
    "Never run out of coins mid-match",
    "Reload now and keep the streak alive",
    "Quick top-ups for serious gamers",
    "Power up your account in one tap",
]


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _NativeFailEnglishSucceedLLM(LLMClient):
    """Stub LLM: native generation fails, English fallback generation succeeds.

    ``complete_json`` keys off the ``system`` argument:

    * ``system == _SYSTEM_PROMPT`` (the standard English/Chinese system prompt
      used by the English fallback generation step) -> returns a healthy batch
      of English candidates.
    * any other system prompt (the native Thai system prompt) -> returns an
      empty candidate list, so post-processing yields zero candidates and the
      attempt is a soft failure. Repeated across all ``_MAX_ATTEMPTS`` this
      exhausts native generation and raises ``GenerationFailureError``.
    """

    def __init__(self, english_candidates: list[str]) -> None:
        self._english_payload = {
            "candidates": [{"copy": c} for c in english_candidates]
        }
        #: Every ``complete_json`` call, recorded as ``{"prompt", "system"}``.
        self.json_calls: list[dict[str, Any]] = []

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: Optional[int] = None,
    ) -> str:  # pragma: no cover - generation uses complete_json only
        return ""

    async def complete_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: Optional[int] = None,
    ) -> dict:
        self.json_calls.append({"prompt": prompt, "system": system})
        if system == _SYSTEM_PROMPT:
            # English fallback generation -> succeed.
            return dict(self._english_payload)
        # Native (Thai) generation -> no candidates -> soft failure.
        return {"candidates": []}

    # -- helpers for assertions -------------------------------------------

    @property
    def native_call_count(self) -> int:
        return sum(1 for c in self.json_calls if c["system"] != _SYSTEM_PROMPT)

    @property
    def english_call_count(self) -> int:
        return sum(1 for c in self.json_calls if c["system"] == _SYSTEM_PROMPT)


class _FakeLocalizationTool:
    """Deterministic Localization_Tool double; records every translate call.

    ``translate`` always succeeds, returning a :class:`LocalizerOutput` whose
    ``localized_versions`` maps each requested target language to a stable,
    source-derived translation (``"[<lang>] <source>"``). This keeps each
    translated candidate unique and lets the test assert the translation path
    ran.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def translate(
        self,
        source_copy: str,
        source_language: Target_Language = Target_Language.EN,
        target_languages: Optional[list[Target_Language]] = None,
        target_market: Target_Market = Target_Market.EN_GLOBAL,
    ) -> LocalizerOutput:
        self.calls.append(
            {
                "source_copy": source_copy,
                "source_language": source_language,
                "target_languages": list(target_languages or []),
                "target_market": target_market,
            }
        )
        localized = {
            lang: f"[{lang.value}] {source_copy}"
            for lang in (target_languages or [])
        }
        return LocalizerOutput(localized_versions=localized, failed_languages=[])


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _make_brief() -> Creative_Brief:
    """A valid Thai-market brief (English source). Native generation applies."""
    return Creative_Brief(
        campaign_topic="Game top-up promotion",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=_MARKET,
        creative_type=Creative_Type.HEADLINE,
        selling_points=["Fast top-ups", "Secure payments"],
        target_audience="Mobile gamers in Thailand",
        source_language="en",
        brand_name="Coco",
    )


def _make_platform_spec() -> Platform_Spec:
    return Platform_Spec(
        platform=Target_Platform.GOOGLE_ADS,
        char_limits={
            Creative_Type.HEADLINE: 90,
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
    )


def _make_generator(
    llm: _NativeFailEnglishSucceedLLM,
    localization_tool: _FakeLocalizationTool,
) -> CreativeGenerator:
    # Small timeout is still ample: native retries add ~0.3s of backoff sleeps
    # and the English fallback succeeds on its first attempt.
    return CreativeGenerator(
        llm,
        timeout_ms=10_000,
        prompt_selector=LanguagePromptSelector(),
        localization_tool=localization_tool,  # type: ignore[arg-type]
    )


def _find_event(records: list[dict], event: str) -> dict:
    """Return the single captured structlog record with the given event name."""
    matches = [r for r in records if r.get("event") == event]
    assert matches, f"expected a {event!r} log record; captured: {records!r}"
    assert len(matches) == 1, f"expected exactly one {event!r} record; got {matches!r}"
    return matches[0]


# ---------------------------------------------------------------------------
# Requirement 1.6 — native failure falls back to English + translate
# ---------------------------------------------------------------------------


class TestNativeGenerationFallbackSucceeds:
    """After 3 failed native attempts, generate() succeeds via the fallback."""

    async def test_generate_succeeds_via_fallback(self) -> None:
        llm = _NativeFailEnglishSucceedLLM(_ENGLISH_CANDIDATES)
        localizer = _FakeLocalizationTool()
        generator = _make_generator(llm, localizer)

        output = await generator.generate(
            _make_brief(),
            _make_platform_spec(),
            min_count=3,
            request_id="req-native-fallback",
        )

        # The request ultimately succeeds despite native generation failing.
        assert len(output.candidates) >= 3

    async def test_candidates_stamped_with_target_language(self) -> None:
        """Translated fallback candidates carry the target generation_language."""
        llm = _NativeFailEnglishSucceedLLM(_ENGLISH_CANDIDATES)
        localizer = _FakeLocalizationTool()
        generator = _make_generator(llm, localizer)

        output = await generator.generate(
            _make_brief(),
            _make_platform_spec(),
            min_count=3,
            request_id="req-native-fallback",
        )

        # Every returned candidate was translated into the market's primary
        # language and re-stamped accordingly (Req 1.6).
        assert output.candidates  # non-empty
        for candidate in output.candidates:
            assert candidate.generation_language == _PRIMARY_LANGUAGE
            # The fake translator prefixes "[th] " — proof the copy is the
            # translated (not the raw English) text.
            assert candidate.source_copy.startswith(f"[{_PRIMARY_LANGUAGE}] ")

    async def test_english_then_translate_path_invoked(self) -> None:
        """Fallback runs English generation then calls the Localization_Tool."""
        llm = _NativeFailEnglishSucceedLLM(_ENGLISH_CANDIDATES)
        localizer = _FakeLocalizationTool()
        generator = _make_generator(llm, localizer)

        output = await generator.generate(
            _make_brief(),
            _make_platform_spec(),
            min_count=3,
            request_id="req-native-fallback",
        )

        # Native generation was attempted exactly _MAX_ATTEMPTS times, then the
        # English fallback generation ran (at least once).
        assert llm.native_call_count == _MAX_ATTEMPTS
        assert llm.english_call_count >= 1

        # The Localization_Tool translate step ran once per returned candidate,
        # each requesting the market's primary language.
        assert len(localizer.calls) == len(output.candidates)
        for call in localizer.calls:
            assert call["source_language"] == Target_Language.EN
            assert call["target_languages"] == [Target_Language.TH]
            assert call["target_market"] == _MARKET


# ---------------------------------------------------------------------------
# Requirement 1.6 — fallback event logging
# ---------------------------------------------------------------------------


class TestNativeGenerationFallbackLogging:
    """The fallback event is logged with target language, reason, and count."""

    async def test_fallback_event_logged_with_required_fields(self) -> None:
        llm = _NativeFailEnglishSucceedLLM(_ENGLISH_CANDIDATES)
        localizer = _FakeLocalizationTool()
        generator = _make_generator(llm, localizer)

        with capture_logs() as records:
            await generator.generate(
                _make_brief(),
                _make_platform_spec(),
                min_count=3,
                request_id="req-native-fallback",
            )

        record = _find_event(records, "creative_generator.native_generation_fallback")
        assert record["log_level"] == "warning"
        # Required fields per Req 1.6.
        assert record["target_language"] == _PRIMARY_LANGUAGE
        assert record["attempt_count"] == _MAX_ATTEMPTS
        # The failure reason is carried through from the GenerationFailureError.
        assert isinstance(record["failure_reason"], str)
        assert record["failure_reason"]
