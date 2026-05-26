"""Diagnostic — does mimo-v2.5 actually return JSON when asked?

Sends 3 different JSON-shaped prompts at increasing complexity to figure
out where the model breaks down. Prints the *raw* HTTP response body so we
can see exactly what came back instead of guessing.

Usage:
    & "C:\\Users\\kangs\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" test_json.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv("TOKENPONY_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("TOKENPONY_API_KEY", "")
MODEL = os.getenv("TOKENPONY_MODEL", "")


async def call(prompt: str, *, max_tokens: int = 512, temperature: float = 0.1) -> dict:
    """Make a raw chat-completions call so we can inspect every byte returned."""
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return r.json()


def show(label: str, payload: dict) -> None:
    """Pretty-print the relevant slice of the API response."""
    print(f"---- {label} ----")
    try:
        content = payload["choices"][0]["message"]["content"]
        finish = payload["choices"][0].get("finish_reason", "?")
        usage = payload.get("usage", {})
        print(f"finish_reason = {finish}")
        print(f"usage         = {usage}")
        print(f"content       = {content!r}")
        print(f"content_len   = {len(content)}")
    except (KeyError, IndexError) as exc:
        print(f"shape error: {exc}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


async def main() -> int:
    if not (BASE_URL and API_KEY and MODEL):
        print("ERROR: missing TOKENPONY_* env vars (run setup_env.py)")
        return 1

    print(f"Model: {MODEL}")
    print(f"Base : {BASE_URL}")
    print()

    # Test 1: bare text — confirms the channel works.
    p1 = "Reply with the single word: PONG"
    show("test 1 — bare text reply", await call(p1, max_tokens=20))

    # Test 2: minimal JSON — easiest possible JSON ask.
    p2 = 'Return ONLY this JSON object on one line: {"ok": true}'
    show("test 2 — minimal JSON literal", await call(p2, max_tokens=64))

    # Test 3: 4-key score JSON — exactly what CTA_Optimizer asks for.
    p3 = (
        "Score the call-to-action quality of: 'Get 20% Bonus on Topup'\n"
        "Return ONLY a JSON object on a single line:\n"
        '{"verb_strength": 0.7, "urgency": 0.6, '
        '"benefit_clarity": 0.8, "cultural_fit": 0.7}'
    )
    show("test 3 — CTA-shaped JSON (the failing case)", await call(p3, max_tokens=512))

    # Test 4: same but with much higher max_tokens, in case the model is
    # silently being cut off mid-output.
    show("test 4 — CTA-shaped JSON, max_tokens=2048", await call(p3, max_tokens=2048))

    # Test 5: same prompt, temperature 0.7 (the model may dislike T=0.1).
    show(
        "test 5 — CTA-shaped JSON, temperature=0.7",
        await call(p3, max_tokens=512, temperature=0.7),
    )

    print("Diagnosis hints:")
    print("  - test 1 OK + test 2/3 empty → model refuses to emit JSON literally")
    print("  - test 3 empty + test 4 non-empty → max_tokens too small")
    print("  - test 3 empty + test 5 non-empty → low temperature triggers empty")
    print("  - all empty → switch to a different model (qwen / deepseek / gpt-4o-mini)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
