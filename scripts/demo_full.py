"""Full end-to-end demo — generation + compliance + localization + embed + CTA + ranking.

This drives the complete Orchestrator flow against the real
TokenPony / mimo-v2.5-pro endpoint. Because each LLM round-trip on this
endpoint costs ~30-80 seconds, the demo deliberately uses a small Brief
(EN_GLOBAL market = 1 language only, 1 keyword, HEADLINE = ≤30 chars) so
the whole run completes in ~3-8 minutes instead of 15+.

Usage:
    & "C:\\Users\\kangs\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" demo_full.py

What this verifies
------------------
* Creative_Generator        — produces ≥5 candidates (LLM call #1, slowest)
* Compliance_Checker        — local dictionary (no LLM) for initial check
* Localization_Tool         — translates to EN only (no-op for EN source)
* Keyword_Embedder          — embeds 1 keyword (LLM call per candidate)
* Compliance_Checker recheck — runs after embed
* CTA_Optimizer             — scores trailing CTA (LLM call per candidate)
* Composite Scorer          — final ranking
* Trace persistence         — `traces/<request_id>/trace.json` written

Key trick
---------
We pass `llm=None` to ComplianceChecker so it runs in dictionary-only mode.
That alone saves ~5 LLM calls and brings the total well under 10 calls.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


async def main() -> int:
    from creative_agent import configure_logging, get_logger
    from creative_agent.api import handle_request
    from creative_agent.llm import RealLLMClient
    from creative_agent.orchestrator import Orchestrator
    from creative_agent.tools.compliance_checker import ComplianceChecker
    from creative_agent.tools.creative_generator import CreativeGenerator
    from creative_agent.tools.cta_optimizer import CTAOptimizer
    from creative_agent.tools.keyword_embedder import KeywordEmbedder
    from creative_agent.tools.localization_tool import LocalizationTool

    # Use WARNING so the structured log lines don't drown the human-readable
    # progress prints below. The trace.json file still captures everything.
    configure_logging(level="WARNING")
    log = get_logger("demo_full")

    try:
        llm = RealLLMClient()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("Did you run `python setup_env.py`?")
        return 1

    # Dictionary-only compliance — saves ~5 LLM calls. The whole point of
    # the local dictionary is exactly this: deterministic BLOCK detection
    # without round-trips.
    compliance_checker = ComplianceChecker(llm=None)
    creative_generator = CreativeGenerator(llm)
    localization_tool = LocalizationTool(llm)
    keyword_embedder = KeywordEmbedder(llm)
    cta_optimizer = CTAOptimizer(llm, compliance_checker=compliance_checker)

    orchestrator = Orchestrator(
        creative_generator=creative_generator,
        compliance_checker=compliance_checker,
        localization_tool=localization_tool,
        keyword_embedder=keyword_embedder,
        cta_optimizer=cta_optimizer,
    )

    # Minimal brief — EN_GLOBAL market keeps localization to a single
    # language (en), 1 keyword keeps embedder fast, HEADLINE keeps copy
    # short.
    brief = {
        "campaign_topic": "Game topup bonus weekend",
        "target_platform": "GOOGLE_ADS",
        "target_market": "EN_GLOBAL",
        "creative_type": "HEADLINE",
        "keywords": ["topup"],
        "selling_points": ["20% bonus", "instant credit"],
        "source_language": "en",
    }

    print("=" * 70)
    print("Full pipeline demo — Creative Editor Agent")
    print("=" * 70)
    print()
    print("Brief:")
    print(json.dumps(brief, indent=2, ensure_ascii=False))
    print()
    print("Calling LLM (this typically takes 3-8 minutes against mimo-v2.5-pro)...")
    print("Watch the printed log lines below — every stage announces when it starts.")
    print()

    t0 = time.perf_counter()
    result = await handle_request(brief, orchestrator)
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 70)
    print(f"DONE in {elapsed:.1f}s — HTTP {result['status_code']}")
    print("=" * 70)

    body = result["body"]

    # Pretty-print: success goes through a curated summary, errors get the
    # raw payload.
    if result["status_code"] == 200:
        print()
        print(f"request_id           = {body['request_id']}")
        print(f"generation_time_ms   = {body['generation_time_ms']}")
        print(f"refill_count         = {body['refill_count']}")
        print(f"total_generated      = {body['total_candidates_generated']}")
        print(f"total_filtered_out   = {body['total_candidates_filtered_out']}")
        print(f"warnings             = {body.get('warnings', [])}")
        print()
        print("Ranked candidates (highest composite_score first):")
        print()
        for rank, c in enumerate(body["ranked_candidates"], 1):
            print(f"  [#{rank}] {c['source_copy']}")
            print(f"        composite_score    = {c['composite_score']:.4f}")
            print(f"        compliance_score   = {c['compliance_report']['compliance_score']:.2f}")
            print(f"        keyword_coverage   = {c['keyword_coverage']:.2f}")
            print(f"        cta_strength_score = {c['cta_strength_score']:.2f}")
            print(f"        hit_keywords       = {c['hit_keywords']}")
            print(f"        skipped_keywords   = {c['skipped_keywords']}")
            if c.get("warnings"):
                print(f"        warnings           = {c['warnings']}")
            print()

        trace_path = ROOT / "traces" / body["request_id"] / "trace.json"
        if trace_path.is_file():
            print(f"Full trace saved to: {trace_path}")
        else:
            print(
                f"(no trace.json found at {trace_path} — this is fine, "
                "the orchestrator writes per-request traces only when "
                "TraceRecorder is wired in)"
            )
    else:
        print()
        print("Error response payload:")
        print(json.dumps(body, indent=2, ensure_ascii=False))

    return 0 if result["status_code"] == 200 else 2


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
