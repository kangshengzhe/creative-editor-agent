"""Review_Translator — translate delivered copies for the HK review team.

Coco is a Hong Kong company; its operators read Chinese. When a campaign
targets, say, Vietnam, the generated copy is Vietnamese — which the reviewer
can't easily vet. This component produces a *comprehension aid*: each delivered
candidate's ``source_copy`` rendered in Simplified Chinese, Traditional
Chinese, and English, shown in the UI detail panel.

This is deliberately separate from ``localized_versions`` (the actual
ad-market translations). The review languages are fixed (operator-facing), not
market-driven, and English is skipped when the copy is already English.

Performance
-----------
All copies of a request are translated in ONE batched ``complete_json`` call
(not per-candidate), so the cost is a single extra LLM round-trip per creative
type rather than dozens. Failures degrade gracefully to an empty map — the
review aid is best-effort and never blocks delivery.
"""

from __future__ import annotations

from typing import Optional

from creative_agent.errors.codes import ToolFailureError
from creative_agent.llm.client import LLMClient
from creative_agent.observability.logging import get_logger

__all__ = ["ReviewTranslator", "REVIEW_LANGUAGES"]

#: Fixed operator-review languages (HK team). ``en`` is dropped when the copy
#: is already English (see :meth:`ReviewTranslator.translate`).
REVIEW_LANGUAGES: tuple[str, ...] = ("zh-Hans", "zh-Hant", "en")

_LLM_MAX_TOKENS: int = 4096
_LLM_TEMPERATURE: float = 0.2
_LLM_TIMEOUT_MS: int = 30000

_SYSTEM_PROMPT: str = (
    "You are a precise translator helping a Hong Kong ad-ops team review "
    "foreign-language ad copy. Translate each provided ad copy into the "
    "requested review languages, preserving meaning and tone. Keep brand / "
    "product names unchanged.\n\n"
    "Review language tags:\n"
    "- zh-Hans = Simplified Chinese\n"
    "- zh-Hant = Traditional Chinese\n"
    "- en = English\n\n"
    "Respond with STRICT JSON only (no markdown, no commentary):\n"
    '{"items": [{"index": <int>, "zh-Hans": "...", "zh-Hant": "...", '
    '"en": "..."}, ...]}\n'
    "Include exactly the requested language keys for every item, in input "
    "order, one entry per input copy."
)


class ReviewTranslator:
    """Batch-translates delivered copies into the HK review languages."""

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
        """Translate ``copies`` into the review languages, one batched call.

        Args:
            copies: The delivered candidate source copies, in order.
            copy_is_english: When True the source is already English, so the
                ``en`` review language is omitted (only zh-Hans / zh-Hant).
            request_id: Optional id for structured logs.

        Returns:
            A list parallel to ``copies``; element ``i`` is a
            ``{lang_tag: translation}`` map for ``copies[i]``. On any failure
            returns a list of empty dicts (best-effort; never raises).
        """
        if not copies:
            return []

        languages = [lang for lang in REVIEW_LANGUAGES if not (lang == "en" and copy_is_english)]
        empty: list[dict[str, str]] = [{} for _ in copies]

        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(copies))
        prompt = (
            f"Review languages requested: {', '.join(languages)}\n\n"
            "Translate each of these ad copies into the requested review "
            "languages:\n"
            f"{numbered}"
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
            return empty
        except Exception as exc:  # noqa: BLE001 — best-effort aid, never block
            self._log.warning(
                "review_translator.failed",
                request_id=request_id,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return empty

        return self._parse(payload, len(copies), languages)

    @staticmethod
    def _parse(
        payload: object, count: int, languages: list[str]
    ) -> list[dict[str, str]]:
        """Map the LLM JSON back to a per-copy list, tolerant of omissions."""
        result: list[dict[str, str]] = [{} for _ in range(count)]
        if not isinstance(payload, dict):
            return result
        items = payload.get("items")
        if not isinstance(items, list):
            return result
        for entry in items:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            if not isinstance(idx, int) or not (0 <= idx < count):
                continue
            langs: dict[str, str] = {}
            for lang in languages:
                val = entry.get(lang)
                if isinstance(val, str) and val.strip():
                    langs[lang] = val.strip()
            if langs:
                result[idx] = langs
        return result
