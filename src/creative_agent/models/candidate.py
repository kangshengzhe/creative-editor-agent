"""Creative_Candidate and CTAVariant models.

Mirrors ``design.md`` → *Data Models / Creative_Candidate*. Score fields are
constrained to ``[0.0, 1.0]`` per Property 3 (validates requirements 3.4,
5.2, 6.3, 7.2).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .compliance import Compliance_Report
from .enums import Creative_Type, Target_Language, Target_Platform


class CTADimensions(BaseModel):
    """Four-dimensional CTA scoring breakdown (requirement 6.4)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    verb_strength: float = Field(..., ge=0.0, le=1.0)
    urgency: float = Field(..., ge=0.0, le=1.0)
    benefit_clarity: float = Field(..., ge=0.0, le=1.0)
    cultural_fit: float = Field(..., ge=0.0, le=1.0)


class CTAVariant(BaseModel):
    """A single CTA candidate produced by CTA_Optimizer (requirement 6.1)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    text: str = Field(..., description="The CTA text itself.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate CTA strength score in [0.0, 1.0].",
    )
    dimensions: CTADimensions = Field(
        ...,
        description="Per-dimension breakdown that produced ``score``.",
    )


class FailedLanguage(BaseModel):
    """Record of a target language whose translation was skipped or failed.

    Used by Localization_Tool when a single language fails (requirement 9.3).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    lang: str = Field(
        ...,
        description=(
            "Language code (typically a ``Target_Language`` value, but "
            "kept as ``str`` so the failure record can also describe "
            "rejected unsupported codes — requirement 4.9)."
        ),
    )
    reason: str = Field(..., description="Human-readable failure reason.")


class Creative_Candidate(BaseModel):
    """A single ad creative candidate flowing through the pipeline."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- Identity ---------------------------------------------------------
    candidate_id: str = Field(..., description="Per-request unique candidate id.")
    generation_index: int = Field(
        ...,
        ge=0,
        description="Generation order within the request; used as final "
        "tie-break key in AB_Ranking (requirement 7.4).",
    )

    # --- Source copy ------------------------------------------------------
    source_copy: str = Field(..., description="Original (English) copy.")
    source_language: str = Field(
        default="en",
        description="Source language; currently always ``en``.",
    )

    # --- Tool outputs -----------------------------------------------------
    compliance_report: Compliance_Report = Field(
        ..., description="Compliance verdict for ``source_copy``."
    )
    keyword_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Hit-keyword ratio in [0.0, 1.0] (requirement 5.2).",
    )
    hit_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that landed inside the final copy.",
    )
    skipped_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that could not be embedded (requirement 5.6).",
    )
    cta_strength_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="CTA strength score in [0.0, 1.0] (requirement 6.3).",
    )
    cta_variants: Optional[list[CTAVariant]] = Field(
        default=None,
        description=(
            "Populated only when ``creative_type == CTA``; carries the "
            "≥5 ranked CTA candidates (requirement 6.1)."
        ),
    )

    # --- Localization -----------------------------------------------------
    localized_versions: dict[Target_Language, str] = Field(
        default_factory=dict,
        description=(
            "Translated copies keyed by ``Target_Language``. Languages "
            "absent from this map either were not requested or failed and "
            "are then listed in ``failed_languages`` (requirement 9.3)."
        ),
    )
    failed_languages: list[FailedLanguage] = Field(
        default_factory=list,
        description="Per-language translation failures (requirement 9.3).",
    )

    # --- Scoring ----------------------------------------------------------
    composite_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Composite score injected by Composite Scorer "
            "(requirement 7.2). Defaults to 0.0 prior to scoring."
        ),
    )

    # --- Localization & diversity metadata --------------------------------
    angle_label: str | None = Field(
        default=None,
        description=(
            "Creative angle this candidate was generated for, assigned by "
            "Angle_Splitter / round-robin generation (requirement 3.6). "
            "``None`` when angle-based generation was not used."
        ),
    )
    generation_language: str = Field(
        default="en",
        description=(
            "Language the candidate copy was generated in; ``en`` for the "
            "standard English flow, otherwise the target market's primary "
            "language for native generation (requirement 1.1)."
        ),
    )
    display_width: int | None = Field(
        default=None,
        description=(
            "Display_Unit width of the copy as computed by "
            "Display_Width_Calculator (requirement 4.8). ``None`` until "
            "display-width enforcement runs."
        ),
    )
    semantic_embedding: list[float] | None = Field(
        default=None,
        exclude=True,
        description=(
            "Sentence-embedding vector used by Semantic_Diversity_Checker "
            "(requirement 2.1). Excluded from serialization."
        ),
    )

    # --- Routing metadata --------------------------------------------------
    target_platform: Target_Platform = Field(
        ..., description="Target ad platform for this candidate."
    )
    creative_type: Creative_Type = Field(
        ..., description="Creative copy type for this candidate."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Tool-level degradation hints (e.g. ``compliance_check_failed`` "
            "from requirement 9.2, ``keyword_embed_failed`` from "
            "requirement 9.4)."
        ),
    )


__all__ = [
    "CTADimensions",
    "CTAVariant",
    "FailedLanguage",
    "Creative_Candidate",
]
