"""Compliance_Report and Violation models.

Mirrors ``design.md`` → *Data Models / Compliance_Report*. The semi-open
interval ``[start, end)`` representation is required by requirement 3.2.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import Compliance_Severity, ViolationCategory


class Violation(BaseModel):
    """A single compliance violation entry within a Compliance_Report.

    Position fields use a semi-open interval ``[start, end)`` (requirement 3.2).
    ``matched_term`` is optional because LLM-based semantic checks may flag
    a passage without a single matched lexicon entry (e.g. exaggeration
    spanning a phrase).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    start: int = Field(
        ...,
        ge=0,
        description="Inclusive start index (UTF-16/code-point — caller decides).",
    )
    end: int = Field(
        ...,
        ge=0,
        description="Exclusive end index. Must be > ``start``.",
    )
    category: ViolationCategory = Field(
        ..., description="Violation category."
    )
    severity: Compliance_Severity = Field(
        ..., description="Violation severity."
    )
    matched_term: Optional[str] = Field(
        default=None,
        description=(
            "Original term that triggered the rule. Optional for semantic "
            "(LLM) findings without a single matched token."
        ),
    )
    suggestion: str = Field(
        ...,
        description="Human-readable remediation suggestion.",
    )

    @model_validator(mode="after")
    def _validate_range(self) -> "Violation":
        if self.end <= self.start:
            raise ValueError(
                f"Violation.end ({self.end}) must be greater than "
                f"Violation.start ({self.start}); intervals are [start, end)."
            )
        return self


class Compliance_Report(BaseModel):
    """Aggregated compliance result for a single piece of copy."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    compliance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Compliance score in [0.0, 1.0]; 1.0 means fully compliant "
            "(requirement 3.9). Scoring rules: empty/whitespace -> 0.0; "
            "any BLOCK -> 0.0; otherwise max(0.1, 1.0 - 0.2 * warn_count)."
        ),
    )
    violations: list[Violation] = Field(
        default_factory=list,
        description="Violation hits; empty when fully compliant.",
    )
    checked_at: str = Field(
        ...,
        description="ISO-8601 timestamp when the check completed.",
    )
    checker_version: str = Field(
        ...,
        description="Version tag of the compliance checker / dictionary set.",
    )


__all__ = ["Violation", "Compliance_Report"]
