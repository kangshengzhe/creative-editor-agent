"""Five core tools: Creative_Generator, Compliance_Checker, Localization_Tool,
Keyword_Embedder, CTA_Optimizer.

This module re-exports the tool classes and their shared output dataclasses
from :mod:`creative_agent.tools.types` so callers can write::

    from creative_agent.tools import LocalizationTool, EmbedderOutput

Imports of the per-tool classes are wrapped in ``try / except ImportError``
because the tools land via parallel sub-agent work; an aggregator that
unconditionally imported every module would break the moment one file is
absent. ``__all__`` is built from the names that actually resolved.
"""

from creative_agent.tools.types import (
    CTAOptimizerOutput,
    EmbedderOutput,
    LocalizerOutput,
)

# Tools may be created by parallel sub-agents in any order; tolerate any
# subset being present at import time.
try:
    from creative_agent.tools.creative_generator import CreativeGenerator  # noqa: F401
except ImportError:  # pragma: no cover — only triggered during partial builds
    pass

try:
    # GeneratorOutput is owned by the sub-agent that delivers task 7.1; it
    # may live in ``types.py`` (added by that sub-agent) or in the
    # ``creative_generator`` module itself. Try ``types`` first, then the
    # tool module, and silently skip if neither exposes it.
    from creative_agent.tools.types import GeneratorOutput  # noqa: F401
except ImportError:
    try:
        from creative_agent.tools.creative_generator import (  # noqa: F401
            GeneratorOutput,
        )
    except ImportError:  # pragma: no cover
        pass

try:
    from creative_agent.tools.compliance_checker import ComplianceChecker  # noqa: F401
except ImportError:  # pragma: no cover
    pass

try:
    from creative_agent.tools.types import CheckerOutput  # noqa: F401
except ImportError:
    try:
        from creative_agent.tools.compliance_checker import (  # noqa: F401
            CheckerOutput,
        )
    except ImportError:  # pragma: no cover
        pass

try:
    from creative_agent.tools.localization_tool import (  # noqa: F401
        LocalizationTool,
        market_to_languages,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from creative_agent.tools.keyword_embedder import KeywordEmbedder  # noqa: F401
except ImportError:  # pragma: no cover
    pass

try:
    from creative_agent.tools.cta_optimizer import CTAOptimizer  # noqa: F401
except ImportError:  # pragma: no cover
    pass


# Build __all__ from names that actually resolved. This keeps wildcard
# imports tidy regardless of which sub-agent delivered which file.
__all__ = [
    name
    for name in (
        "CreativeGenerator",
        "GeneratorOutput",
        "ComplianceChecker",
        "CheckerOutput",
        "LocalizationTool",
        "market_to_languages",
        "KeywordEmbedder",
        "CTAOptimizer",
        "LocalizerOutput",
        "EmbedderOutput",
        "CTAOptimizerOutput",
    )
    if name in globals()
]
