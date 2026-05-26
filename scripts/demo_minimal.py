"""Minimal end-to-end demo — generation + compliance only.

Skips Localization, Keyword_Embedder, and CTA_Optimizer to minimise the
number of LLM round-trips, since mimo-v2.5-pro on TokenPony costs ~30-50s
per call. With this stripped-down flow you should see a result in ~1-3
minutes instead of ~5-15.

Usage:
    python demo_minimal.py

What it does
------------
1. Validate a sample Creative_Brief.
2. Call Creative_Generator → receive 5-7 raw candidates.
3. For each candidate, call Compliance_Checker once → attach the report.
4. Print every candidate with its compliance score and any violation hits.

What it skips
-------------
* Localization (would multiply LLM calls by 2-3x)
* Keyword_Embedder (per-candidate LLM call)
* CTA_Optimizer (per-candidate LLM call)
* Composite Scorer ranking (no point ranking on partial data)
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
    from creative_agent.config import load_platform_spec
    from creative_agent.llm import RealLLMClient
    from creative_agent.models import Creative_Brief, Target_Language
    from creative_agent.tools.compliance_checker import ComplianceChecker
    from creative_agent.tools.creative_generator import CreativeGenerator

    configure_logging(level="INFO")
    log = get_logger("demo")

    try:
        llm = RealLLMClient()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    creative_generator = CreativeGenerator(llm, timeout_ms=180000)
    compliance_checker = ComplianceChecker(llm=None)  # dictionary-only, no LLM

    # Build a tiny Brief — fewer keywords, simpler topic, faster prompt.
    brief = Creative_Brief.model_validate({
        "campaign_topic": "Game topup bonus",
        "target_platform": "GOOGLE_ADS",
        "target_market": "EN_GLOBAL",
        "creative_type": "HEADLINE",
        "keywords": ["topup"],
        "source_language": "en",
    })
    platform_spec = load_platform_spec(brief.target_platform)

    print("=" * 70)
    print("Minimal pipeline demo — Creative_Generator + Compliance_Checker")
    print("=" * 70)
    print(f"Topic: {brief.campaign_topic}")
    print(f"Platform: {brief.target_platform.value}  Market: {brief.target_market.value}")
    print(f"Type: {brief.creative_type.value}  (char limit: {platform_spec.char_limit(brief.creative_type)})")
    print()

    # --- Stage 1: generate candidates ----------------------------------
    print("[1/2] Calling Creative_Generator (this is the slowest step) ...")
    t0 = time.perf_counter()
    try:
        gen_output = await creative_generator.generate(
            brief=brief,
            platform_spec=platform_spec,
            min_count=5,
            request_id="demo_min_001",
        )
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 2

    gen_elapsed = time.perf_counter() - t0
    print(f"  OK — got {len(gen_output.candidates)} candidates in {gen_elapsed:.1f}s")
    print()

    for i, c in enumerate(gen_output.candidates, 1):
        print(f"  [{i}] {c.source_copy}")
    print()

    # --- Stage 2: compliance check (dictionary-only, very fast) --------
    print("[2/2] Running Compliance_Checker on each candidate (local dictionary, instant) ...")
    t1 = time.perf_counter()
    reports = await asyncio.gather(*[
        compliance_checker.check(c.source_copy, Target_Language.EN)
        for c in gen_output.candidates
    ])
    check_elapsed = time.perf_counter() - t1
    print(f"  OK — {len(reports)} reports in {check_elapsed:.2f}s")
    print()

    # --- Result table --------------------------------------------------
    print("=" * 70)
    print("Result")
    print("=" * 70)
    output: list[dict] = []
    for c, report in zip(gen_output.candidates, reports):
        entry = {
            "copy": c.source_copy,
            "len": len(c.source_copy),
            "compliance_score": report.compliance_score,
            "violations": [
                {
                    "category": v.category.value,
                    "severity": v.severity.value,
                    "matched_term": v.matched_term,
                    "span": [v.start, v.end],
                }
                for v in report.violations
            ],
        }
        output.append(entry)

    print(json.dumps(output, indent=2, ensure_ascii=False))

    print()
    print(f"Total time: {gen_elapsed + check_elapsed:.1f}s "
          f"(generation {gen_elapsed:.1f}s + compliance {check_elapsed:.2f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
