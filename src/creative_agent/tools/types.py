"""Shared output dataclasses for the five core tools.

This module hosts the lightweight dataclass envelopes that wrap each tool's
result so the orchestrator can pass them around without taking a hard
dependency on the tools' implementation modules. Each ``*Output`` mirrors
the corresponding section of ``design.md`` § Components and Interfaces.

The dataclasses are intentionally framework-free (no Pydantic, no
``BaseModel``): they are plain Python ``@dataclass`` envelopes whose fields
already point at fully-validated Pydantic models (e.g. ``Compliance_Report``,
``CTAVariant``, ``FailedLanguage``). Validation belongs with the persistent
domain models in :mod:`creative_agent.models`; these envelopes are the
ephemeral wire format between tools and the orchestrator.

Notes:
    * ``GeneratorOutput`` and ``CheckerOutput`` are owned by the parallel
      sub-agent implementing tasks 7.1 and 8.1; they are *not* defined here
      to avoid two sub-agents trying to write the same symbol. The
      ``__init__.py`` aggregator uses ``try / except ImportError`` to
      tolerate either ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from creative_agent.models import (
    CTAVariant,
    FailedLanguage,
    Target_Language,
)

__all__ = [
    "LocalizerOutput",
    "EmbedderOutput",
    "CTAOptimizerOutput",
]


@dataclass
class LocalizerOutput:
    """Output of :class:`creative_agent.tools.LocalizationTool`.

    Mirrors design.md § 5. Localization_Tool. Languages that succeeded land
    in :attr:`localized_versions`; per-language failures (LLM error,
    placeholder mismatch, timeout) are captured under
    :attr:`failed_languages` so the orchestrator can apply the
    Requirement 9.3 degradation policy without aborting the candidate.
    """

    localized_versions: dict[Target_Language, str] = field(default_factory=dict)
    failed_languages: list[FailedLanguage] = field(default_factory=list)
    localize_time_ms: int = 0


@dataclass
class EmbedderOutput:
    """Output of :class:`creative_agent.tools.KeywordEmbedder`.

    Mirrors design.md § 6. Keyword_Embedder. ``failure_reason`` is populated
    only when the tool could not embed any keyword while keeping the copy
    inside the platform-spec character limit (Requirement 5.9).
    """

    embedded_copy: str
    keyword_coverage: float
    hit_keywords: list[str] = field(default_factory=list)
    skipped_keywords: list[str] = field(default_factory=list)
    embed_time_ms: int = 0
    failure_reason: Optional[str] = None


@dataclass
class CTAOptimizerOutput:
    """Output of :class:`creative_agent.tools.CTAOptimizer`.

    Mirrors design.md § 7. CTA_Optimizer. ``cta_variants`` is non-``None``
    only when the candidate's ``creative_type`` is ``CTA`` (Requirement 6.1);
    for the other creative types only the trailing CTA segment is scored
    (Requirement 6.2) and the field stays ``None``.
    """

    cta_strength_score: float
    cta_variants: Optional[list[CTAVariant]] = None
    optimize_time_ms: int = 0
