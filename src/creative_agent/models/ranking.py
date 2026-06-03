"""AB_Ranking — final ordered output of the agent.

Mirrors ``design.md`` → *Data Models / AB_Ranking*. The ordering invariant
itself (Property 4) is enforced by Composite Scorer; this model is the
serialization-friendly container that carries the result back to callers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .candidate import Creative_Candidate


class AB_Ranking(BaseModel):
    """Final ranked candidate set returned to the caller (requirement 7.1)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(
        ...,
        description="Per-request id assigned by the API Gateway (requirement 1.7).",
    )
    brief_summary: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Compact echo of the originating brief — typically "
            "``{'topic': ..., 'platform': ..., 'market': ...}``. Kept as "
            "a free-form dict to avoid coupling the response shape to the "
            "full Creative_Brief schema."
        ),
    )
    ranked_candidates: list[Creative_Candidate] = Field(
        default_factory=list,
        description=(
            "Candidates sorted by composite_score desc, then "
            "compliance_score desc, then cta_strength_score desc, then "
            "generation_index asc (requirement 7.4)."
        ),
    )
    total_candidates_generated: int = Field(
        ...,
        ge=0,
        description="Total candidates produced by Creative_Generator across "
        "the initial run plus any refill rounds.",
    )
    total_candidates_filtered_out: int = Field(
        ...,
        ge=0,
        description="Number of candidates dropped due to BLOCK violations "
        "(requirement 7.5).",
    )
    generation_time_ms: int = Field(
        ...,
        ge=0,
        description="End-to-end generation time in milliseconds "
        "(requirement 8.1 budget is 15000 ms).",
    )
    refill_count: int = Field(
        ...,
        ge=0,
        le=2,
        description="Number of refill rounds triggered. 0, 1, or 2 "
        "(requirement 7.6).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Request-level warnings (e.g. keyword truncation "
        "from requirement 1.6, partial-language failures, etc.).",
    )
    angle_distribution: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Mapping of each angle label to the number of candidates "
            "generated for that angle (requirement 3.6). Empty when "
            "angle-based generation was not used."
        ),
    )
    diversity_multiplier: float = Field(
        default=1.0,
        description=(
            "Applied diversity multiplier for the candidate set: "
            "``1.0 + 0.05 * (distinct_angles - 1)``, capped at a maximum "
            "of 1.25 (requirement 3.7). Defaults to 1.0 when no diversity "
            "bonus applies."
        ),
    )
    target_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Ad_Group_Quota target for this request's creative_type "
            "(requirement 5.7): HEADLINE=15, DESCRIPTION=10, CTA=5, "
            "LONG_COPY=5, or the brief's explicit override. Lets consumers "
            "detect under-fill by comparing against ``len(ranked_candidates)``. "
            "Defaults to 0 for callers that do not set it."
        ),
    )


__all__ = ["AB_Ranking"]
