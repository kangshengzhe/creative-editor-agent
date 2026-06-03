"""Property-based tests for the Angle_Splitter.

Feature: creative-localization-diversity.

Exercises the universal correctness properties from design.md for
:class:`creative_agent.integration.angle_splitter.AngleSplitter`. Each property
is tagged with its design property number and the requirement(s) it validates.

The LLM is mocked with :class:`creative_agent.llm.mock_client.MockLLMClient`,
configured to return a valid JSON decomposition whose angle count is derived
from the number of selling points (always >= the splitter minimum, and
sometimes above the maximum) so the splitter's clamping/inference logic is
exercised across the full input space.
"""

from __future__ import annotations

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from creative_agent.integration.angle_splitter import AngleSplitter
from creative_agent.llm.mock_client import MockLLMClient

#: Angle labels drawn from the splitter's predefined taxonomy; the test builds
#: realistic, distinct angles by cycling through these and suffixing an index.
_TAXONOMY = (
    "convenience",
    "price",
    "speed",
    "safety",
    "quality",
    "trust",
    "exclusivity",
    "social proof",
)


def _build_angles_payload(count: int) -> dict:
    """Return a valid ``{"angles": [...]}`` payload with ``count`` distinct angles."""
    angles = []
    for i in range(count):
        label = _TAXONOMY[i % len(_TAXONOMY)]
        # Suffix beyond the first cycle keeps labels distinct and realistic.
        if i >= len(_TAXONOMY):
            label = f"{label} {i}"
        angles.append(
            {
                "label": label,
                "description": f"Lead with {label}.",
                "source": "selling_point" if i % 2 == 0 else "inferred",
            }
        )
    return {"angles": angles}


# Feature: creative-localization-diversity, Property 6: Angle decomposition count bounds
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    selling_points=st.lists(st.text(), min_size=0, max_size=10),
    campaign_topic=st.text(),
    target_audience=st.one_of(st.none(), st.text()),
)
def test_angle_decomposition_count_bounds(
    selling_points: list[str],
    campaign_topic: str,
    target_audience: str | None,
) -> None:
    """For any selling points (incl. null/empty) + topic + audience, the
    Angle_Splitter returns between 4 and 8 angles inclusive.

    The mock LLM returns ``len(selling_points) + 4`` angles, which ranges from
    4 (no selling points) to 14 (ten selling points). This always meets the
    minimum (so decomposition succeeds without retry) while frequently
    exceeding the maximum, exercising the splitter's upper-bound clamping.

    Validates: Requirements 3.1, 3.2, 3.3
    """
    # Build the mock inside the test body to avoid Hypothesis fixture issues.
    llm = MockLLMClient()
    llm.set_default_response(_build_angles_payload(len(selling_points) + 4))

    splitter = AngleSplitter(llm)

    angles = asyncio.run(
        splitter.decompose(
            selling_points,
            campaign_topic,
            target_audience,
        )
    )

    assert 4 <= len(angles) <= 8
