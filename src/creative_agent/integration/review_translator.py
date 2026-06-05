"""Review_Translator — translate delivered copies for the HK review team.

Coco is a Hong Kong company; its operators read Chinese. When a campaign
targets, say, Vietnam, the generated copy is Vietnamese — which the reviewer
can't easily vet. This component produces a *comprehension aid*: each delivered
candidate's ``source_copy`` rendered in Simplified Chinese, Traditional
Chinese, and English, shown in the UI detail panel.

This is deliberately separate from ``localized_versions`` (the actual
ad-market translations). The review languages are fixed (operator-facing), not
market-driven, and English is skipped when the copy is already English.

Performance & correctness
-------------------------
Each delivered copy is translated by its OWN LLM call, all fired concurrently
via ``asyncio.gather``. We deliberately do NOT batch many copies into one
numbered-list call: with a batched list the model occasionally mislabels or
shifts the per-item ``index`` (and reliably drops the trailing item), which
caused a translation to be attached to the WRONG candidate (or the last item to
be left blank). One copy per call makes misalignment impossible — the result
for copy ``i`` can only come from copy ``i``'s own call. Concurrency keeps the
wall-clock cost close to a single round-trip. Each call is best-effort: a
failure leaves that copy's map empty and never blocks delivery.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from creative_agent.errors.codes import ToolFailureError
from creative_agent.llm.client import LLMClient
from creative_agent.observability.logging import get_logger

__all__ = ["ReviewTranslator", "REVIEW_LANGUAGES"]

#: Fixed operator-review languages (HK team). ``en`` is dropped when the copy
#: is already English (see :meth:`ReviewTranslator.translate`).
REVIEW_LANGUAGES: tuple[str, ...] = ("zh-Hans", "zh-Hant", "en")

_LLM_MAX_TOKENS: int = 1024
_LLM_TEMPERATURE: float = 0.2
_LLM_TIMEOUT_MS: int = 30000

_SYSTEM_PROMPT: str = (
    "You are a precise translator helping a Hong Kong ad-ops team review "
    "foreign-language ad copy. Translate the ONE ad copy the user gives you "
    "into the requested review languages, preserving meaning and tone. Keep "
    "brand / product names unchanged. Translate EXACTLY the text given, even "
    "if it looks truncated or incomplete — do not complete, expand, or guess "
    "missing words.\n\n"
    "Review language tags:\n"
    "- zh-Hans = Simplified Chinese\n"
    "- zh-Hant = Traditional Chinese\n"
    "- en = English\n\n"
    "Respond with STRICT JSON only (no markdown, no commentary), containing "
    "exactly the requested language keys:\n"
    '{"zh-Hans": "...", "zh-Hant": "...", "en": "..."}'
)


class ReviewTranslator:
    """Translates delivered copies into the HK review languages (per-copy)."""

    def __init__(self, llm_client: LLMClient, *, timeout_ms: int = _LLM_TIMEOUT_MS) -> None:
        self._llm = llm_client
        self._timeout_ms = timeout_ms
        self._log = get_logger(__name__)

    async def translate(
        self,
        copies: list[str],
        *,
        copy_is_english: bool,
        request_id: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Translate ``copies`` into the review languages, one call per copy.

        Every copy is translated by its own concurrent LLM call, so the result
        for copy ``i`` is guaranteed to be the translation of ``copies[i]`` (no
        batch index to mislabel) and reflects EXACTLY the delivered — i.e.
        possibly truncated — text the frontend shows.

        Args:
            copies: The delivered candidate source copies, in order.
            copy_is_english: When True the source is already English, so the
                ``en`` review language is omitted (only zh-Hans / zh-Hant).
            request_id: Optional id for structured logs.

        Returns:
            A list parallel to ``copies``; element ``i`` is a
            ``{lang_tag: translation}`` map for ``copies[i]``. A per-copy
            failure yields an empty dict for that copy (best-effort; never
            raises).
        """
        if not copies:
            return []

        languages = [
            lang for lang in REVIEW_LANGUAGES if not (lang == "en" and copy_is_english)
        ]

        results = await asyncio.gather(
            *[
                self._translate_one(copy, languages, request_id=request_id)
                for copy in copies
            ]
        )
        return list(results)

    async def _translate_one(
        self,
        copy: str,
        languages: list[str],
        *,
        request_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Translate a single copy into ``languages``; empty dict on failure."""
        if not copy or not copy.strip():
            return {}

        prompt = (
            f"Review languages requested: {', '.join(languages)}\n\n"
            "Translate this exact ad copy (do not complete or expand it):\n"
            f"{copy}"
        )

        try:
            payload = await self._llm.complete_json(
                prompt,
                system=_SYSTEM_PROMPT,
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE,
                timeout_ms=self._timeout_ms,
            )
        except ToolFailureError as exc:
            self._log.warning(
                "review_translator.failed",
                request_id=request_id,
                reason=exc.message,
            )
            return {}
        except Exception as exc:  # noqa: BLE001 — best-effort aid, never block
            self._log.warning(
                "review_translator.failed",
                request_id=request_id,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return {}

        return self._parse_one(payload, languages)

    @staticmethod
    def _parse_one(payload: object, languages: list[str]) -> dict[str, str]:
        """Extract the requested language keys from a single-copy JSON object."""
        if not isinstance(payload, dict):
            return {}
        langs: dict[str, str] = {}
        for lang in languages:
            val = payload.get(lang)
            if isinstance(val, str) and val.strip():
                langs[lang] = val.strip()
        return langs
