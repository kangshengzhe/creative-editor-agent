"""Example-based unit tests for the single-prompt fallback on angle failure.

Feature: creative-localization-diversity (task 9.4).

These tests cover the angle-decomposition fallback semantics described in
Requirement 3.8: when the :class:`~creative_agent.integration.angle_splitter.AngleSplitter`
fails to decompose the brief (it raises a
:class:`~creative_agent.errors.codes.ToolFailureError` after exhausting its own
retries),
:meth:`~creative_agent.tools.creative_generator.CreativeGenerator.generate_with_angles`
falls back to the existing single-prompt :meth:`generate` path, appends an
angle-decomposition-failure warning to the request-level ``warnings`` list, and
logs the ``creative_generator.angle_decomposition_fallback`` event.

The complementary case (no Angle_Splitter wired) must delegate to single-prompt
generation *without* appending a warning, since angle generation was simply
never requested for that deployment.

The collaborators are injected so the tests are fully deterministic and require
no real LLM / ML dependency:

* A minimal stub Angle_Splitter whose async ``decompose`` always raises a
  ``ToolFailureError`` (``tool_name="Angle_Splitter"``), reproducing the
  post-retry failure of the real splitter without exercising its retry loop.
* A :class:`~creative_agent.llm.mock_client.MockLLMClient` configured with a
  healthy ``{"candidates": [...]}`` batch as its default response, so the
  single-prompt fallback ``generate`` succeeds with enough unique copies to
  clear ``min_count``.

An English-primary market (``EN_GLOBAL``) is used so the single-prompt fallback
runs the straightforward English generation flow (no native-language switching
or translation), keeping the test focused on the angle-fallback behaviour.

The project runs with ``asyncio_mode = "auto"`` (pytest-asyncio), so plain
``async def test_...`` coroutines are awaited automatically. Structured logs are
captured with ``structlog.testing.capture_logs``.
"""

from __future__ import annotations

from typing import Optional

from structlog.testing import capture_logs

from creative_agent.errors.codes import ToolFailureError
from creative_agent.llm.mock_client import MockLLMClient
from creative_agent.models.brief import Creative_Brief
from creative_agent.models.enums import (
    Creative_Type,
    Target_Market,
    Target_Platform,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.tools.creative_generator import CreativeGenerator

# English-primary market so the single-prompt fallback uses the standard
# English generation flow (no native switching / translation).
_MARKET = Target_Market.EN_GLOBAL

# Seven distinct, short English copies the mock returns for the single-prompt
# fallback generation. Enough unique candidates to survive dedup and clear
# min_count comfortably.
_FALLBACK_CANDIDATES = [
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


class _AlwaysFailingAngleSplitter:
    """Stub Angle_Splitter whose ``decompose`` always raises ToolFailureError.

    Reproduces the real splitter's post-retry failure (Requirement 3.8) without
    running its retry loop. The generator only calls ``decompose`` on the
    injected splitter, so this minimal surface is sufficient.
    """

    def __init__(self, message: str = "Angle decomposition failed after 3 attempt(s)") -> None:
        self._message = message
        #: Recorded ``decompose`` invocations as ``(selling_points, topic, audience)``.
        self.calls: list[tuple[object, object, object]] = []

    async def decompose(
        self,
        selling_points: Optional[list[str]],
        campaign_topic: str,
        target_audience: Optional[str],
    ) -> list:
        self.calls.append((selling_points, campaign_topic, target_audience))
        raise ToolFailureError(
            tool_name="Angle_Splitter",
            message=self._message,
            details={"attempts": 3, "min_angles": 4},
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_brief() -> Creative_Brief:
    """A valid English-global brief. Angle decomposition will fail for it."""
    return Creative_Brief(
        campaign_topic="Game top-up promotion",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=_MARKET,
        creative_type=Creative_Type.HEADLINE,
        selling_points=["Fast top-ups", "Secure payments"],
        target_audience="Mobile gamers worldwide",
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


def _make_mock_llm() -> MockLLMClient:
    """A mock LLM whose default response is a healthy single-prompt batch."""
    llm = MockLLMClient()
    llm.set_default_response(
        {"candidates": [{"copy": c} for c in _FALLBACK_CANDIDATES]}
    )
    return llm


def _make_generator(
    llm: MockLLMClient,
    angle_splitter: Optional[object],
) -> CreativeGenerator:
    return CreativeGenerator(
        llm,
        timeout_ms=10_000,
        angle_splitter=angle_splitter,  # type: ignore[arg-type]
    )


def _find_event(records: list[dict], event: str) -> dict:
    """Return the single captured structlog record with the given event name."""
    matches = [r for r in records if r.get("event") == event]
    assert matches, f"expected a {event!r} log record; captured: {records!r}"
    assert len(matches) == 1, f"expected exactly one {event!r} record; got {matches!r}"
    return matches[0]


# ---------------------------------------------------------------------------
# Requirement 3.8 — angle failure falls back to single-prompt + warning
# ---------------------------------------------------------------------------


class TestSinglePromptFallbackOnAngleFailure:
    """When AngleSplitter fails, generation falls back to single-prompt."""

    async def test_fallback_returns_candidates(self) -> None:
        """The request still succeeds via the single-prompt fallback path."""
        llm = _make_mock_llm()
        splitter = _AlwaysFailingAngleSplitter()
        generator = _make_generator(llm, splitter)

        warnings: list[str] = []
        output = await generator.generate_with_angles(
            _make_brief(),
            _make_platform_spec(),
            min_count=5,
            request_id="req-angle-fallback",
            warnings=warnings,
        )

        # The Angle_Splitter was actually consulted (and failed).
        assert len(splitter.calls) == 1
        # Fallback single-prompt generation produced enough unique candidates.
        assert len(output.candidates) >= 5

    async def test_fallback_appends_decomposition_failure_warning(self) -> None:
        """The request-level warnings record the angle-decomposition fallback."""
        llm = _make_mock_llm()
        splitter = _AlwaysFailingAngleSplitter()
        generator = _make_generator(llm, splitter)

        warnings: list[str] = []
        await generator.generate_with_angles(
            _make_brief(),
            _make_platform_spec(),
            min_count=5,
            request_id="req-angle-fallback",
            warnings=warnings,
        )

        # Exactly one warning, indicating angle decomposition failed and the
        # generator fell back to single-prompt generation (Req 3.8).
        assert len(warnings) == 1
        warning = warnings[0]
        assert "angle decomposition failed" in warning
        assert "single-prompt" in warning

    async def test_fallback_logs_decomposition_fallback_event(self) -> None:
        """The angle_decomposition_fallback event is logged on failure."""
        llm = _make_mock_llm()
        splitter = _AlwaysFailingAngleSplitter()
        generator = _make_generator(llm, splitter)

        warnings: list[str] = []
        with capture_logs() as records:
            await generator.generate_with_angles(
                _make_brief(),
                _make_platform_spec(),
                min_count=5,
                request_id="req-angle-fallback",
                warnings=warnings,
            )

        record = _find_event(records, "creative_generator.angle_decomposition_fallback")
        assert record["log_level"] == "warning"
        assert record["request_id"] == "req-angle-fallback"
        # The failure reason is carried through from the ToolFailureError.
        assert isinstance(record["failure_reason"], str)
        assert record["failure_reason"]


# ---------------------------------------------------------------------------
# Requirement 3.8 — no Angle_Splitter wired => silent single-prompt delegation
# ---------------------------------------------------------------------------


class TestNoAngleSplitterDelegatesSilently:
    """No Angle_Splitter wired delegates to single-prompt without a warning."""

    async def test_delegates_to_single_prompt_without_warning(self) -> None:
        llm = _make_mock_llm()
        generator = _make_generator(llm, angle_splitter=None)

        warnings: list[str] = []
        output = await generator.generate_with_angles(
            _make_brief(),
            _make_platform_spec(),
            min_count=5,
            request_id="req-no-splitter",
            warnings=warnings,
        )

        # Single-prompt generation ran and produced candidates.
        assert len(output.candidates) >= 5
        # No decomposition-failure warning: angle generation was never requested.
        assert warnings == []

    async def test_no_decomposition_fallback_event_logged(self) -> None:
        llm = _make_mock_llm()
        generator = _make_generator(llm, angle_splitter=None)

        warnings: list[str] = []
        with capture_logs() as records:
            await generator.generate_with_angles(
                _make_brief(),
                _make_platform_spec(),
                min_count=5,
                request_id="req-no-splitter",
                warnings=warnings,
            )

        events = [r.get("event") for r in records]
        assert "creative_generator.angle_decomposition_fallback" not in events
