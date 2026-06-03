"""Property-based tests for ranking diversity (Composite Scorer).

Feature: creative-localization-diversity

Hypothesis-driven properties validating the diversity-related behaviour of
``creative_agent.orchestrator.composite_scorer`` against the design's
correctness properties (see design.md § Correctness Properties / Testing
Strategy):

* Property 9  — Diversity multiplier formula (Requirement 3.7)
* Property 14 — Angle label attribution completeness (Requirement 3.6)
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from creative_agent.orchestrator.composite_scorer import (
    compute_diversity_multiplier,
)


# Feature: creative-localization-diversity, Property 9: Diversity multiplier formula
@settings(max_examples=100)
@given(distinct_angles=st.integers(min_value=1, max_value=8))
def test_diversity_multiplier_formula(distinct_angles: int) -> None:
    """Property 9: Diversity multiplier formula.

    For any candidate set covering K distinct angles (1 <= K <= 8), the
    diversity multiplier applied to the composite score equals
    ``min(1.0 + 0.05 * (K - 1), 1.25)``. The expected value is computed from
    the formula itself (not a hardcoded table) so the test stays in lock-step
    with the documented requirement.

    **Validates: Requirements 3.7**
    """
    expected = min(1.0 + 0.05 * (distinct_angles - 1), 1.25)
    assert compute_diversity_multiplier(distinct_angles) == expected


# ---------------------------------------------------------------------------
# Property 14 — Angle label attribution completeness
# ---------------------------------------------------------------------------
#
# Reuse the candidate factory from the composite-scorer unit tests rather than
# hand-rolling a second one, so the construction of a valid Creative_Candidate
# (including the required compliance_report / target_platform / creative_type
# shape) stays in one place.
from collections import Counter

from tests.unit.test_composite_scorer import _make_candidate
from creative_agent.orchestrator.composite_scorer import rank_candidates

# A small set of distinct angle labels drawn from the design's angle taxonomy
# (convenience, price, speed, safety, quality, trust, exclusivity, social
# proof). Candidates produced via angle-based generation always carry one of
# these as a non-null ``angle_label``.
_ANGLE_TAXONOMY = (
    "convenience",
    "price",
    "speed",
    "safety",
    "quality",
    "trust",
    "exclusivity",
    "social_proof",
)


# Feature: creative-localization-diversity, Property 14: Angle label attribution completeness
@settings(max_examples=100)
@given(labels=st.lists(st.sampled_from(_ANGLE_TAXONOMY), min_size=4, max_size=8))
def test_angle_label_attribution_completeness(labels: list[str]) -> None:
    """Property 14: Angle label attribution completeness.

    For any candidate set produced via angle-based generation (each candidate
    assigned a non-null angle label), every ranked candidate retains a non-null
    ``angle_label`` and the ranking's ``angle_distribution`` contains a count
    for each distinct angle. The per-label counts match the labels assigned and
    sum to the number of (non-blocked) candidates.

    The generated candidates carry no BLOCK-severity violations, so none are
    filtered out by ``rank_candidates`` — the surviving set equals the input
    set and the distribution sums to ``len(labels)``.

    **Validates: Requirements 3.6**
    """
    candidates = []
    for index, label in enumerate(labels):
        candidate = _make_candidate(generation_index=index)
        # Every candidate gets a non-null angle label (angle-based generation).
        candidate.angle_label = label
        candidates.append(candidate)

    ranking = rank_candidates(
        candidates,
        request_id="test-property-14",
        total_generated=len(candidates),
    )

    # No BLOCK violations were generated, so every candidate survives.
    assert len(ranking.ranked_candidates) == len(labels)

    # (a) Every ranked candidate has a non-null angle_label.
    assert all(c.angle_label is not None for c in ranking.ranked_candidates)

    # (b) angle_distribution keys == the set of labels used, and the per-label
    #     counts match exactly.
    expected_distribution = dict(Counter(labels))
    assert ranking.angle_distribution == expected_distribution
    assert set(ranking.angle_distribution) == set(labels)

    # (c) The distribution counts sum to the number of non-blocked candidates.
    assert sum(ranking.angle_distribution.values()) == len(labels)
