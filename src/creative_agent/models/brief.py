"""Creative_Brief — the request payload submitted by operators.

Mirrors the TypeScript interface declared in ``design.md`` → *Data Models*.

Notes
-----
* Required fields use ``Field`` constraints to enforce length and non-empty
  semantics described by requirements 1.2 and 1.9.
* ``keywords`` is *not* truncated here. Per the task brief, the API Gateway is
  responsible for truncating to 20 entries (requirement 1.6) and emitting the
  user-visible warning. Models should remain pure data containers.
* All optional fields default to ``None`` or empty collections so the model
  can be constructed from a minimal valid payload.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import Creative_Type, Target_Market, Target_Platform


class CampaignPeriod(BaseModel):
    """Optional campaign window. Dates are kept as ISO-8601 strings to avoid
    forcing callers to parse timezones at the API boundary."""

    model_config = ConfigDict(extra="forbid")

    start: str = Field(..., description="ISO-8601 start date/time (inclusive).")
    end: str = Field(..., description="ISO-8601 end date/time (inclusive).")


class Creative_Brief(BaseModel):
    """Structured creative request input.

    Required fields (requirement 1.2):
        * ``campaign_topic``
        * ``target_platform``
        * ``target_market``
        * ``creative_type``
    """

    model_config = ConfigDict(
        extra="forbid",
        # Allow constructing from either enum value or its string form
        # (e.g. ``"GOOGLE_ADS"``) — this is what the JSON gateway delivers.
        use_enum_values=False,
        validate_assignment=True,
    )

    # --- Required fields ---------------------------------------------------
    campaign_topic: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Campaign theme / topic. 1-200 characters (requirement 1.9). "
            "Whitespace-only values are rejected by the API Gateway "
            "(requirement 1.2)."
        ),
    )
    target_platform: Target_Platform = Field(
        ..., description="Target ad platform (requirement 1.3)."
    )
    target_market: Target_Market = Field(
        ..., description="Target market (requirement 1.4)."
    )
    creative_type: Creative_Type = Field(
        ..., description="Creative copy type (requirement 1.5)."
    )

    # --- Optional fields ---------------------------------------------------
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "SEO keywords. The model layer accepts any length; the API "
            "Gateway truncates to the first 20 entries (requirement 1.6)."
        ),
    )
    selling_points: Optional[list[str]] = Field(
        default=None,
        description="Bullet-style selling points to inform copy generation.",
    )
    target_audience: Optional[str] = Field(
        default=None,
        description="Free-form description of the target audience persona.",
    )
    budget: Optional[float] = Field(
        default=None,
        description="Campaign budget; unit is up to the caller.",
    )
    source_language: str = Field(
        default="en",
        description=(
            "Source language of the brief. Defaults to ``en`` because all "
            "tools are anchored on English source copy (design § Tools)."
        ),
    )
    brand_name: Optional[str] = Field(
        default=None,
        description="Brand or product name to preserve through translation.",
    )
    campaign_period: Optional[CampaignPeriod] = Field(
        default=None,
        description="Optional campaign run window.",
    )


__all__ = ["Creative_Brief", "CampaignPeriod"]
