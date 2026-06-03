"""Integration components for the Creative Localization & Diversity feature.

Hosts the four new pure/lightweight components wired into the existing
Orchestrator pipeline:

* :mod:`creative_agent.integration.display_width` — Display_Width_Calculator
* :mod:`creative_agent.integration.language_prompts` — Language_Prompt_Selector
* :mod:`creative_agent.integration.semantic_diversity` — Semantic_Diversity_Checker
* :mod:`creative_agent.integration.angle_splitter` — Angle_Splitter

These modules are placed under ``creative_agent.integration`` so they are
importable through the installable ``creative_agent`` package (the design
document refers to ``src/auto_posting/integration/``; see the spec's
tasks.md Overview for the path reconciliation).
"""

from creative_agent.integration.angle_splitter import Angle, AngleSplitter
from creative_agent.integration.display_width import DisplayWidthCalculator
from creative_agent.integration.keyword_localizer import KeywordLocalizer
from creative_agent.integration.language_prompts import LanguagePromptSelector
from creative_agent.integration.semantic_diversity import (
    DiversityResult,
    EmbeddingUnavailableError,
    SemanticDiversityChecker,
)

__all__ = [
    "Angle",
    "AngleSplitter",
    "DisplayWidthCalculator",
    "KeywordLocalizer",
    "LanguagePromptSelector",
    "DiversityResult",
    "EmbeddingUnavailableError",
    "SemanticDiversityChecker",
]
