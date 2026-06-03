"""Example-based unit tests for the Angle_Splitter fallback path.

Feature: creative-localization-diversity (task 6.3).

These tests cover the failure semantics described in design.md
(§ Error Handling / Angle Splitter Failures) and Requirement 3.8: when the
:class:`~creative_agent.integration.angle_splitter.AngleSplitter` cannot
produce at least ``min_angles`` distinct angles after 3 attempts (1 initial +
2 retries) -- whether because the LLM keeps returning too few angles or because
it returns invalid JSON -- ``decompose`` raises a
:class:`~creative_agent.errors.codes.ToolFailureError`.

The caller (Creative_Generator) catches this exception, falls back to
single-prompt generation, and surfaces a request-level "angle decomposition
failure" warning. These tests assert on the signal the caller relies on: the
error's ``tool_name`` (``"Angle_Splitter"``), its message, and its structured
``details``.

The project runs with ``asyncio_mode = "auto"`` (pytest-asyncio), so plain
``async def test_...`` coroutines are awaited automatically.
"""

from __future__ import annotations

import pytest

from creative_agent.errors.codes import ToolFailureError
from creative_agent.integration.angle_splitter import AngleSplitter, _MAX_ATTEMPTS
from creative_agent.llm.mock_client import MockLLMClient

# A representative, valid brief. The content is irrelevant to the fallback
# behaviour -- the mock LLM ignores it and returns whatever is configured.
_TOPIC = "Same-day grocery delivery"
_AUDIENCE = "Busy urban professionals"
_SELLING_POINTS = ["Fast delivery", "Low prices"]


def _too_few_angles_payload() -> dict:
    """A structurally valid ``{"angles": [...]}`` payload with only 2 angles.

    Two angles is below the default ``min_angles`` of 4, so every attempt is a
    soft failure that drives the splitter through all of its retries.
    """
    return {
        "angles": [
            {
                "label": "speed",
                "description": "Lead with how fast delivery is.",
                "source": "selling_point",
            },
            {
                "label": "price",
                "description": "Lead with affordability.",
                "source": "selling_point",
            },
        ]
    }


def _assert_decomposition_failure(exc: ToolFailureError) -> None:
    """Assert the error carries the angle-decomposition-failure signal.

    This is exactly the signal the caller turns into a request-level warning.
    """
    # tool_name identifies the failing tool for the caller's fallback logic.
    assert exc.tool_name == "Angle_Splitter"
    # The public details dict mirrors tool_name (set by ToolFailureError).
    assert exc.details.get("tool_name") == "Angle_Splitter"
    # The message must clearly indicate angle decomposition failed.
    assert "Angle decomposition failed" in exc.message
    # Structured details record how many attempts were spent and the minimum
    # that could not be satisfied, plus the last underlying error.
    assert exc.details.get("attempts") == _MAX_ATTEMPTS
    assert exc.details.get("min_angles") == 4
    assert "last_error" in exc.details


class TestAngleSplitterTooFewAnglesFallback:
    """Consistently returning fewer than ``min_angles`` -> ToolFailureError."""

    async def test_raises_tool_failure_when_too_few_angles(self) -> None:
        llm = MockLLMClient()
        llm.set_default_response(_too_few_angles_payload())
        splitter = AngleSplitter(llm)

        with pytest.raises(ToolFailureError) as exc_info:
            await splitter.decompose(_SELLING_POINTS, _TOPIC, _AUDIENCE)

        _assert_decomposition_failure(exc_info.value)
        # The last error should explain the too-few-angles soft failure.
        assert "angle" in (exc_info.value.details.get("last_error") or "")

    async def test_llm_invoked_exactly_max_attempts_on_too_few_angles(self) -> None:
        llm = MockLLMClient()
        llm.set_default_response(_too_few_angles_payload())
        splitter = AngleSplitter(llm)

        with pytest.raises(ToolFailureError):
            await splitter.decompose(_SELLING_POINTS, _TOPIC, _AUDIENCE)

        # 1 initial attempt + 2 retries == 3 LLM calls, all requesting JSON.
        assert len(llm.calls) == _MAX_ATTEMPTS
        assert all(call["is_json"] for call in llm.calls)


class TestAngleSplitterInvalidJsonFallback:
    """Invalid JSON from the LLM follows the same failure path."""

    async def test_raises_tool_failure_when_json_invalid(self) -> None:
        llm = MockLLMClient()
        # A non-JSON string makes the mock's complete_json raise a
        # ToolFailureError(tool_name="LLMClient"), which the splitter treats as
        # a retryable failure.
        llm.set_default_response("not valid json {{{")
        splitter = AngleSplitter(llm)

        with pytest.raises(ToolFailureError) as exc_info:
            await splitter.decompose(_SELLING_POINTS, _TOPIC, _AUDIENCE)

        # The raised error is the splitter's own failure, not the LLM's: the
        # caller keys off tool_name == "Angle_Splitter" to surface the warning.
        _assert_decomposition_failure(exc_info.value)

    async def test_llm_invoked_exactly_max_attempts_on_invalid_json(self) -> None:
        llm = MockLLMClient()
        llm.set_default_response("not valid json {{{")
        splitter = AngleSplitter(llm)

        with pytest.raises(ToolFailureError):
            await splitter.decompose(_SELLING_POINTS, _TOPIC, _AUDIENCE)

        assert len(llm.calls) == _MAX_ATTEMPTS

    async def test_empty_selling_points_still_fails_on_invalid_json(self) -> None:
        """Null/empty selling points take the same fallback path on bad JSON."""
        llm = MockLLMClient()
        llm.set_default_response("<<not json>>")
        splitter = AngleSplitter(llm)

        with pytest.raises(ToolFailureError) as exc_info:
            await splitter.decompose(None, _TOPIC, _AUDIENCE)

        _assert_decomposition_failure(exc_info.value)
