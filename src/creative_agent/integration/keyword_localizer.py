"""Keyword_Localizer — localize generic SEO keywords, keep brand terms verbatim.

Why this exists
---------------
SEO keywords were previously forced to appear *verbatim in English* in every
candidate, regardless of the target market's language. So a keyword like
``topup`` stayed English even in a Spanish (Uruguay) or Arabic ad, while the
rest of the copy was localized — which reads wrong and, per PPC localization
best practice, hurts relevance and conversion.

Industry guidance (verified 2026-06, multiple PPC/localization sources) splits
keywords into a tiered glossary:

* **Non-translatable** — brand names, product names, SKUs, proper nouns, and
  English tech terms with no natural local equivalent → keep verbatim.
* **Translatable** — generic concept words (e.g. "topup" = "recharge") → adapt
  to the term local users actually search in their language.

Word-for-word translation of *whole ad copy* underperforms, but generic
keywords should still be expressed in the local language; only brand/proper
terms stay fixed. This module asks the shared LLM to make exactly that
distinction and returns an ``original -> localized`` mapping.

Behaviour
---------
* English-primary targets (or an empty keyword list) → identity mapping, no
  LLM call.
* Otherwise one structured ``complete_json`` call returns, per keyword, the
  localized form (or the original verbatim when it's a brand/proper noun).
* Any failure (LLM error, bad JSON, missing keys) degrades gracefully to the
  identity mapping (keep the original keyword) so a localization hiccup never
  drops keywords or fails the request.
"""

from __future__ import annotations

from typing import Optional

from creative_agent.errors.codes import ToolFailureError
from creative_agent.llm.client import LLMClient
from creative_agent.models.enums import Target_Language
from creative_agent.observability.logging import get_logger

__all__ = ["KeywordLocalizer"]

_LLM_MAX_TOKENS: int = 1024
_LLM_TEMPERATURE: float = 0.2  # low — this is a precise lexical task
_LLM_TIMEOUT_MS: int = 30000

#: English code — these targets keep keywords exactly as supplied (no call).
_ENGLISH_CODE: str = Target_Language.EN.value

_SYSTEM_PROMPT: str = (
    "You are a senior multilingual PPC (paid search) specialist. Your job is "
    "to localize SEO keywords for ad copy in a specific target language, "
    "following standard keyword-localization practice.\n\n"
    "For each keyword decide one of two things:\n"
    "1. TRANSLATE it to the word local users actually search in the target "
    "language, if it is a generic concept (e.g. English 'topup' -> Spanish "
    "'recarga'; 'bonus' -> 'bono'). Use the natural, commonly-searched local "
    "term, not a literal dictionary gloss.\n"
    "2. KEEP it verbatim (unchanged) if it is a brand name, product name, SKU, "
    "proper noun, or an English technical term with no natural local "
    "equivalent.\n\n"
    "Respond with STRICT JSON only (no markdown, no commentary):\n"
    '{"keywords": [{"original": "<input>", "localized": "<output>", '
    '"kept_verbatim": true|false}, ...]}\n'
    "Return one entry per input keyword, in the same order."
)


class KeywordLocalizer:
    """Maps SEO keywords to their target-language form (brand terms kept)."""

    def __init__(self, llm_client: LLMClient, *, timeout_ms: int = _LLM_TIMEOUT_MS) -> None:
        self._llm = llm_client
        self._timeout_ms = timeout_ms
        self._log = get_logger(__name__)

    async def localize(
        self,
        keywords: list[str],
        target_language: str,
        *,
        request_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Return an ``{original: localized}`` map for ``keywords``.

        For an English target or empty keyword list this is the identity map
        and makes no LLM call. On any failure it falls back to identity so
        keywords are never lost.

        Args:
            keywords: The brief's SEO keywords (already truncated to <= 20).
            target_language: Primary :class:`Target_Language` code of the
                market (e.g. ``"es"``, ``"ar"``). ``"en"`` -> identity.
            request_id: Optional id for structured logs.

        Returns:
            ``{original_keyword: localized_keyword}``. Always contains every
            input keyword as a key.
        """
        cleaned = [k for k in keywords if k and k.strip()]
        identity = {k: k for k in cleaned}

        # English target or nothing to do → keep keywords exactly as given.
        if not cleaned or target_language.strip().lower() == _ENGLISH_CODE:
            return identity

        prompt = (
            f"Target language code: {target_language}\n"
            "Localize these SEO keywords (translate generic terms, keep brand/"
            "proper nouns verbatim):\n"
            + "\n".join(f"- {k}" for k in cleaned)
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
                "keyword_localizer.fallback_identity",
                request_id=request_id,
                target_language=target_language,
                reason=exc.message,
            )
            return identity
        except Exception as exc:  # noqa: BLE001 — never fail the request over this
            self._log.warning(
                "keyword_localizer.fallback_identity",
                request_id=request_id,
                target_language=target_language,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return identity

        mapping = self._parse(payload, cleaned)
        self._log.info(
            "keyword_localizer.localized",
            request_id=request_id,
            target_language=target_language,
            mapping=mapping,
        )
        return mapping

    @staticmethod
    def _parse(payload: object, originals: list[str]) -> dict[str, str]:
        """Build the mapping from the LLM JSON, defaulting to identity.

        Tolerant: any entry that is missing, malformed, or has an empty
        ``localized`` falls back to the original keyword. Originals not
        mentioned by the model also default to themselves, so the returned map
        always covers every input.
        """
        result: dict[str, str] = {k: k for k in originals}
        if not isinstance(payload, dict):
            return result
        entries = payload.get("keywords")
        if not isinstance(entries, list):
            return result

        # Match returned entries to originals case-insensitively.
        by_lower = {k.lower(): k for k in originals}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            original = entry.get("original")
            localized = entry.get("localized")
            if not isinstance(original, str):
                continue
            key = by_lower.get(original.strip().lower())
            if key is None:
                continue
            if isinstance(localized, str) and localized.strip():
                result[key] = localized.strip()
        return result
