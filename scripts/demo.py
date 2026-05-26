"""End-to-end demo for the Creative Editor Agent.

Usage:
    python demo.py

Reads LLM credentials from `.env` (set up via `python setup_env.py`).
Generates 5 ad creative candidates for a sample Brief, runs the full
pipeline (compliance check + localization + keyword embed + CTA optimize),
ranks them, and prints the result as JSON.

Expected output: an `AB_Ranking` with 3-5 ranked candidates, each carrying:
- source_copy
- compliance_report (score + violations)
- localized_versions (fil + en for the PH market)
- keyword_coverage + hit_keywords
- cta_strength_score
- composite_score (the headline ranking metric)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make sure src/ is on the path even if the package isn't installed yet.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


async def main() -> int:
    # Imports are inside main so a missing dependency surfaces with a clear
    # error message rather than a bare ImportError at module load time.
    from creative_agent import configure_logging, get_logger
    from creative_agent.api import handle_request
    from creative_agent.llm import RealLLMClient
    from creative_agent.orchestrator import Orchestrator
    from creative_agent.tools.compliance_checker import ComplianceChecker
    from creative_agent.tools.creative_generator import CreativeGenerator
    from creative_agent.tools.cta_optimizer import CTAOptimizer
    from creative_agent.tools.keyword_embedder import KeywordEmbedder
    from creative_agent.tools.localization_tool import LocalizationTool

    configure_logging(level="INFO")
    log = get_logger("demo")

    # 1. Wire the LLM client (reads TOKENPONY_* from .env).
    try:
        llm = RealLLMClient()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("Did you run `python setup_env.py`?")
        return 1

    # 2. Wire the five tools. ComplianceChecker is shared between the
    #    pipeline (initial / recheck) and CTA_Optimizer (post-filter).
    compliance_checker = ComplianceChecker(llm=llm)
    creative_generator = CreativeGenerator(llm)
    localization_tool = LocalizationTool(llm)
    keyword_embedder = KeywordEmbedder(llm)
    cta_optimizer = CTAOptimizer(llm, compliance_checker=compliance_checker)

    # 3. Wire the orchestrator.
    orchestrator = Orchestrator(
        creative_generator=creative_generator,
        compliance_checker=compliance_checker,
        localization_tool=localization_tool,
        keyword_embedder=keyword_embedder,
        cta_optimizer=cta_optimizer,
    )

    # 4. Build a sample Brief — Diwali topup bonus, PH market, Google Ads
    #    headline, with three SEO keywords.
    brief = {
        "campaign_topic": "Diwali topup bonus for casual mobile gamers",
        "target_platform": "GOOGLE_ADS",
        "target_market": "PH",
        "creative_type": "HEADLINE",
        "keywords": ["topup", "Diwali", "bonus"],
        "selling_points": [
            "20% bonus on first topup",
            "instant credit to game wallet",
            "limited 7-day promo window",
        ],
        "target_audience": "Filipino mobile gamers ages 18-34",
        "brand_name": "Coco",
        "source_language": "en",
    }

    log.info("demo.start", brief=brief)
    print("=" * 70)
    print("Submitting Brief to Creative Editor Agent...")
    print("=" * 70)
    print(json.dumps(brief, indent=2, ensure_ascii=False))
    print()

    result = await handle_request(brief, orchestrator)

    print("=" * 70)
    print(f"Response (HTTP {result['status_code']}):")
    print("=" * 70)
    print(json.dumps(result["body"], indent=2, ensure_ascii=False))
    return 0 if result["status_code"] == 200 else 2


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
