"""Single-shot LLM connectivity check.

Sends one tiny `complete()` request to the configured TokenPony / MiMo
endpoint to verify that:
1. .env was loaded correctly
2. The API key is accepted
3. The base URL is reachable from this machine
4. The model name is valid

Usage: python test_llm.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


async def main() -> int:
    from creative_agent.llm import RealLLMClient

    # Read config (key is masked when printed) so the user sees what's
    # actually being used.
    api_key = os.getenv("TOKENPONY_API_KEY", "")
    base_url = os.getenv("TOKENPONY_BASE_URL", "")
    model = os.getenv("TOKENPONY_MODEL", "")
    masked = (api_key[:4] + "..." + api_key[-4:]) if len(api_key) > 8 else "(too short)"

    print("Configuration loaded from .env:")
    print(f"  TOKENPONY_API_KEY  = {masked}")
    print(f"  TOKENPONY_BASE_URL = {base_url}")
    print(f"  TOKENPONY_MODEL    = {model}")
    print()

    if not api_key or not base_url or not model:
        print("ERROR: One or more values missing. Run `python setup_env.py`.")
        return 1

    try:
        client = RealLLMClient()
    except ValueError as exc:
        print(f"ERROR constructing RealLLMClient: {exc}")
        return 1

    prompt = "Reply with the single word: PONG"
    print(f"Sending tiny prompt to {model} (timeout 60s)...")
    print(f"  prompt = {prompt!r}")
    print()

    start = time.perf_counter()
    try:
        reply = await client.complete(
            prompt,
            max_tokens=200,
            temperature=0.0,
            timeout_ms=60000,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"FAILED after {elapsed:.1f}s")
        print(f"  error type: {type(exc).__name__}")
        print(f"  error msg : {exc}")
        return 2

    elapsed = time.perf_counter() - start
    print(f"OK in {elapsed:.1f}s")
    print(f"  reply: {reply!r}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
