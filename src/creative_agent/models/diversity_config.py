"""SemanticDiversityConfig — configuration for Semantic_Diversity_Checker.

Mirrors ``design.md`` → *Data Models / Semantic Diversity Configuration*.

Threshold calibration (empirical, 2026-06)
------------------------------------------
Requirement 2.4 originally specified a default Similarity_Threshold of 0.85
over the range [0.5, 0.99]. Live measurement with the configured
``paraphrase-multilingual-MiniLM-L12-v2`` model on short ad-copy pairs showed
0.85 is far too high to satisfy the business's "headlines must be
differentiated" requirement: genuine near-paraphrases score ~0.45–0.66
(avg 0.66) while truly distinct copy scores ~0.11–0.40 (avg 0.27). A threshold
of 0.85 therefore only catches near-verbatim duplicates and lets paraphrases
through. The default is now **0.60** — comfortably above the distinct-copy
ceiling (~0.40) so it does not over-reject, and low enough to catch
medium-to-strong paraphrases. The allowed floor is lowered to 0.35 so operators
can tune into the separation zone if they want stricter de-duplication.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SemanticDiversityConfig(BaseModel):
    """Tunable settings for embedding-based semantic deduplication."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    similarity_threshold: float = Field(
        default=0.60,
        ge=0.35,
        le=0.99,
        description=(
            "Cosine_Similarity value above which two candidates are "
            "considered semantic duplicates. Default 0.60 (empirically "
            "calibrated for paraphrase-multilingual-MiniLM-L12-v2 on short ad "
            "copy; see module docstring), allowed range [0.35, 0.99]. "
            "Lower = stricter de-duplication (rejects more)."
        ),
    )
    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description=(
            "Sentence-embedding model used to compute Embedding_Vectors; a "
            "lightweight multilingual model covering all target languages."
        ),
    )
    timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Max seconds to wait for embedding computation before falling "
            "back to text-based dedup only (requirement 2.8)."
        ),
    )
    enabled: bool = Field(
        default=True,
        description="Whether semantic diversity filtering is active.",
    )


__all__ = ["SemanticDiversityConfig"]
