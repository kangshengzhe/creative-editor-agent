"""Composite Scorer — composite_score formula and AB_Ranking ordering.

Implements design.md § Components / 8. Composite Scorer and Requirements
7.1 — 7.5, 7.8.

Composite score formula (Req 7.2)::

    composite_score = 0.5 * compliance_score
                    + 0.25 * keyword_coverage
                    + 0.25 * cta_strength_score

The ``compliance_score`` input is read from
``candidate.compliance_report.compliance_score`` — that is the canonical
source produced by Compliance_Checker. ``Creative_Candidate`` does not
carry a top-level ``compliance_score`` field; only the report does.

Sort key (Req 7.4) — three-level tie-break + generation-order fallback::

    (-composite_score,
     -compliance_score,
     -cta_strength_score,
     generation_index)

BLOCK filtering (Req 7.5) — any candidate whose Compliance_Report contains
at least one violation with ``severity == BLOCK`` is dropped before sorting.

This module is pure: no LLM calls, no filesystem access, no state.
The Orchestrator owns lifecycle concerns; this file owns the math.
"""

from __future__ import annotations

from typing import Any, Optional

from creative_agent.models import (
    AB_Ranking,
    Compliance_Severity,
    Creative_Candidate,
)
from creative_agent.observability.logging import get_logger

__all__ = [
    "compute_composite_score",
    "compute_angle_distribution",
    "compute_diversity_multiplier",
    "rank_candidates",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Composite score (Requirement 7.2)
# ---------------------------------------------------------------------------

# Weights are baked in deliberately — Req 7.2 fixes them and Property 3
# verifies the exact formula. Exposing them as parameters would be a footgun.
_W_COMPLIANCE: float = 0.5
_W_KEYWORD: float = 0.25
_W_CTA: float = 0.25


def compute_composite_score(candidate: Creative_Candidate) -> float:
    """Return the composite score for ``candidate`` per Requirement 7.2.

    The formula reads ``compliance_score`` from
    ``candidate.compliance_report.compliance_score`` (the only place it
    lives — see ``models/candidate.py``). Because each component score is
    bounded in ``[0, 1]`` and the weights sum to 1.0, the result is also
    bounded in ``[0, 1]`` — no clamping needed in the happy path. We keep
    a defensive clamp anyway so any tiny floating-point drift cannot trip
    the ``validate_assignment=True`` constraint when the value is written
    back to ``candidate.composite_score``.

    Args:
        candidate: The candidate whose three component scores are read.

    Returns:
        ``0.5 * compliance_score + 0.25 * keyword_coverage
        + 0.25 * cta_strength_score`` clamped to ``[0.0, 1.0]``.
    """
    compliance_score = candidate.compliance_report.compliance_score
    score = (
        _W_COMPLIANCE * compliance_score
        + _W_KEYWORD * candidate.keyword_coverage
        + _W_CTA * candidate.cta_strength_score
    )
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


# ---------------------------------------------------------------------------
# BLOCK filtering (Requirement 7.5)
# ---------------------------------------------------------------------------


def _has_block(candidate: Creative_Candidate) -> bool:
    """Return ``True`` iff any violation in the report is BLOCK severity."""
    return any(
        v.severity == Compliance_Severity.BLOCK
        for v in candidate.compliance_report.violations
    )


# ---------------------------------------------------------------------------
# Diversity multiplier & angle distribution (Requirements 3.6, 3.7)
# ---------------------------------------------------------------------------

# Per Requirement 3.7 the multiplier grows by 0.05 for each distinct angle
# beyond the first and is capped at 1.25 (the cap is reached at 6 distinct
# angles). Property 9 verifies the exact formula.
_DIVERSITY_STEP: float = 0.05
_DIVERSITY_CAP: float = 1.25


def compute_angle_distribution(
    candidates: list[Creative_Candidate],
) -> dict[str, int]:
    """Return a mapping of angle label → candidate count (Requirement 3.6).

    Only candidates carrying a non-null ``angle_label`` contribute. When no
    candidate has an angle label the mapping is empty, signalling that
    angle-based generation was not used (existing behaviour is preserved).

    Args:
        candidates: The candidate set whose angle labels are tallied.

    Returns:
        ``{angle_label: count}`` for every distinct non-null label present.
    """
    distribution: dict[str, int] = {}
    for candidate in candidates:
        label = candidate.angle_label
        if label is None:
            continue
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def compute_diversity_multiplier(distinct_angles: int) -> float:
    """Return the diversity multiplier for ``distinct_angles`` (Req 3.7).

    Formula::

        min(1.0 + 0.05 * (distinct_angles - 1), 1.25)

    When there are zero distinct angles (angle-based generation was not
    used) the multiplier is 1.0, leaving composite scores unchanged.

    Args:
        distinct_angles: Number of distinct non-null angle labels covered
            by the candidate set.

    Returns:
        The diversity multiplier in ``[1.0, 1.25]``.
    """
    if distinct_angles <= 0:
        return 1.0
    multiplier = 1.0 + _DIVERSITY_STEP * (distinct_angles - 1)
    if multiplier > _DIVERSITY_CAP:
        return _DIVERSITY_CAP
    return multiplier


# ---------------------------------------------------------------------------
# Ranking (Requirements 7.1, 7.3, 7.4, 7.5, 7.8)
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: list[Creative_Candidate],
    request_id: str,
    refill_count: int = 0,
    generation_time_ms: int = 0,
    warnings: Optional[list[str]] = None,
    brief_summary: Optional[dict[str, Any]] = None,
    total_generated: Optional[int] = None,
    target_count: int = 0,
) -> AB_Ranking:
    """Filter, score, sort and package candidates into an :class:`AB_Ranking`.

    Steps, in order:

    1. **Filter** out every candidate carrying a BLOCK violation
       (Requirement 7.5). The number of dropped candidates is recorded in
       ``total_candidates_filtered_out``.
    2. **Score** each survivor by writing the composite score back to
       ``candidate.composite_score`` (Requirement 7.8). Candidates use
       ``validate_assignment=True``, but ``composite_score`` is provably
       bounded in ``[0, 1]`` so the assignment cannot raise.
    3. **Sort** by the four-key tuple
       ``(-composite, -compliance, -cta_strength, generation_index)`` —
       the negatives make the first three keys descending while
       ``generation_index`` ascends, satisfying Requirement 7.4.
    4. **Package** into an :class:`AB_Ranking`.

    Args:
        candidates: Pipeline output candidates (post-pipeline, pre-rank).
            May be empty; the resulting ranking will simply have no
            ``ranked_candidates``.
        request_id: Per-request id from the API gateway (Req 1.7).
        refill_count: Number of refill rounds executed (0, 1 or 2 —
            Req 7.6). Defaults to 0.
        generation_time_ms: End-to-end generation time, measured by the
            Orchestrator. Defaults to 0 for callers that don't track it.
        warnings: Request-level warnings (e.g. keyword truncation from
            Req 1.6, partial-language failures). Defaults to ``[]``.
        brief_summary: Compact echo of the originating brief — typically
            ``{"topic", "platform", "market"}``. Stored verbatim on the
            ranking. Defaults to ``{}``.
        total_generated: Total candidates produced across the initial run
            plus any refill rounds, *before* BLOCK filtering. Used to
            populate ``total_candidates_generated`` and to compute
            ``total_candidates_filtered_out``. When omitted, defaults to
            ``len(candidates)`` (i.e. assumes the caller already passed
            the full pre-filter set).

    Returns:
        :class:`AB_Ranking` with the candidates sorted in A/B-ready order.
    """
    pre_filter_count = (
        total_generated if total_generated is not None else len(candidates)
    )

    survivors: list[Creative_Candidate] = [
        c for c in candidates if not _has_block(c)
    ]

    # Inject the base composite score onto each survivor.
    for candidate in survivors:
        candidate.composite_score = compute_composite_score(candidate)

    survivors.sort(
        key=lambda c: (
            -c.composite_score,
            -c.compliance_report.compliance_score,
            -c.cta_strength_score,
            c.generation_index,
        )
    )

    # ------------------------------------------------------------------
    # Ad-group quota truncation (Requirement 5 / platform cap)
    # ------------------------------------------------------------------
    # Angle-based generation produces candidates in batches of 3 per angle,
    # so the surviving count often overshoots the Target_Count (e.g. 6 angles
    # x 3 = 18 for a 15-headline target). Google Ads RSA ad groups cap at 15
    # headlines / 10 descriptions, so we keep only the highest-scoring
    # ``target_count`` candidates — the surplus served as an optimise-then-
    # select pool. ``target_count == 0`` (callers that don't set it) disables
    # truncation, preserving the prior return-everything behaviour.
    overshoot = 0
    if target_count > 0 and len(survivors) > target_count:
        overshoot = len(survivors) - target_count
        survivors = survivors[:target_count]

    # ------------------------------------------------------------------
    # Diversity bonus (Requirements 3.6, 3.7)
    # ------------------------------------------------------------------
    # Tally the angle labels present in the surviving set and derive the
    # uniform diversity multiplier. The multiplier is applied *after*
    # sorting: because it is a single positive factor shared by every
    # survivor it preserves the relative ordering established above, so
    # the ranked order is unaffected while each reported ``composite_score``
    # reflects the diversity bonus. When no candidate carries an angle
    # label the distribution is empty and the multiplier is 1.0, leaving
    # existing (non-angle) behaviour completely intact. Computed on the
    # post-truncation set so the reported distribution matches what is
    # actually delivered.
    angle_distribution = compute_angle_distribution(survivors)
    distinct_angles = len(angle_distribution)
    diversity_multiplier = compute_diversity_multiplier(distinct_angles)

    if diversity_multiplier != 1.0:
        for candidate in survivors:
            boosted = candidate.composite_score * diversity_multiplier
            # ``composite_score`` is constrained to [0, 1]; clamp before
            # assignment so ``validate_assignment=True`` cannot reject the
            # boosted value.
            candidate.composite_score = boosted if boosted < 1.0 else 1.0

    # ``filtered_out`` counts only BLOCK-violation drops (Requirement 7.5).
    # Add back ``overshoot`` so the quota truncation above is not miscounted
    # as compliance filtering: (generated - delivered) - quota_trimmed.
    filtered_out = pre_filter_count - len(survivors) - overshoot
    if filtered_out < 0:
        # Defensive: caller passed an inconsistent ``total_generated``
        # (e.g. a number smaller than the candidates list). Clamp to
        # zero so the downstream ``ge=0`` validator on AB_Ranking
        # doesn't reject the response.
        filtered_out = 0

    ranking = AB_Ranking(
        request_id=request_id,
        brief_summary=dict(brief_summary) if brief_summary else {},
        ranked_candidates=survivors,
        total_candidates_generated=pre_filter_count,
        total_candidates_filtered_out=filtered_out,
        generation_time_ms=generation_time_ms,
        refill_count=refill_count,
        warnings=list(warnings) if warnings else [],
        angle_distribution=angle_distribution,
        diversity_multiplier=diversity_multiplier,
        target_count=target_count,
    )

    log.info(
        "composite_scorer.ranked",
        request_id=request_id,
        total_generated=pre_filter_count,
        ranked_count=len(survivors),
        filtered_out=filtered_out,
        truncated_overshoot=overshoot,
        target_count=target_count,
        refill_count=refill_count,
        distinct_angles=distinct_angles,
        diversity_multiplier=diversity_multiplier,
    )
    return ranking
