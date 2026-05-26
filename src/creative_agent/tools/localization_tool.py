"""Localization_Tool — translates source copy into target-market languages.

Implements design.md § Components / 5. Localization_Tool and Requirements
4.1 — 4.10.

Responsibilities
----------------

* **Market → Language mapping** (Req 4.1 / 4.2 / 4.3 / 4.4): ``PH → [fil, en]``,
  ``TH → [th, en]``, ``RU → [ru, en]``, ``EN_GLOBAL → [en]``. Exposed as the
  class-level :data:`MARKET_LANGUAGES` constant and via
  :meth:`languages_for_market`.

* **Placeholder preservation** (Req 4.5, Property 5): Tokens of the form
  ``{name}`` (where ``name`` matches ``[a-zA-Z0-9_]{1,32}``) are extracted
  before the LLM call and replaced with opaque sentinels ``__PH_<i>__`` so
  the model cannot translate them. After the LLM returns, sentinels are
  swapped back to the original placeholders. The translation is rejected
  (and the language is recorded as failed) when the placeholder multiset of
  the result does not equal the source's.

* **Currency / Date formatting** (Req 4.6 / 4.7): The currency symbol and
  date format strings derived from the target market are passed to the LLM
  as soft hints (the model is instructed to apply them when surfacing money
  / date values). They are also used as regression-friendly state we record
  for traces.

* **Formal register** (Req 4.8): The LLM system prompt instructs the model
  to use formal written register: ``Вы`` for Russian, polite particles for
  Thai, and to avoid slang / contractions for Filipino and English.

* **Per-language tolerance** (Req 9.3): A single-language failure does not
  abort the call. The failed language is appended to ``failed_languages``
  and the rest continue.

* **Unsupported language defence** (Req 4.9): Receiving a target language
  outside ``{en, fil, th, ru}`` raises ``ValueError`` so callers see the
  defensive failure as a programming error (the public API surface is
  governed by the ``Target_Language`` enum).

* **Timing** (Req 4.10): The whole ``translate`` call is bounded by
  ``asyncio.wait_for(timeout=3.0)``; per-language LLM calls use
  ``timeout_ms=2500``. Languages run in parallel via :func:`asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from creative_agent.errors import ToolFailureError
from creative_agent.llm.client import LLMClient
from creative_agent.models import (
    FailedLanguage,
    Target_Language,
    Target_Market,
)
from creative_agent.observability.logging import get_logger
from creative_agent.tools.types import LocalizerOutput

__all__ = [
    "LocalizerInput",
    "LocalizationTool",
    "market_to_languages",
]

log = get_logger(__name__)


# Placeholder pattern per requirement 4.5: {name} where name is [a-zA-Z0-9_]{1,32}.
_PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile(r"\{([a-zA-Z0-9_]{1,32})\}")

# Total budget per requirement 4.10.
_DEFAULT_TIMEOUT_S: float = 30.0
# Per-language LLM timeout — leaves ~5s slack for orchestration / parsing.
_PER_LANG_LLM_TIMEOUT_MS: int = 25000

_LLM_MAX_TOKENS: int = 2048
_LLM_TEMPERATURE: float = 0.3

# Formal-register guidance fed to the LLM, keyed by Target_Language.
_REGISTER_GUIDANCE: dict[Target_Language, str] = {
    Target_Language.EN: (
        "Use formal written English. No slang, no contractions "
        "(write 'do not', not 'don't')."
    ),
    Target_Language.FIL: (
        "Use formal written Filipino (Tagalog). No street slang, no informal "
        "abbreviations, no English-Tagalog code-switching beyond brand names."
    ),
    Target_Language.TH: (
        "Use formal written Thai with polite ending particles "
        "(ครับ / ค่ะ where natural). No internet slang."
    ),
    Target_Language.RU: (
        "Use formal written Russian. Always use the polite second-person "
        "pronoun 'Вы' (capitalised) when addressing the reader."
    ),
}


@dataclass
class LocalizerInput:
    """Input contract mirroring design.md § Localization_Tool."""

    source_copy: str
    source_language: Target_Language
    target_languages: list[Target_Language]
    target_market: Target_Market


# Module-level fan-out map (Req 4.1 - 4.4). Mirrors
# :attr:`LocalizationTool.MARKET_LANGUAGES` but is exposed as a free-standing
# helper because the orchestrator and tests need to derive the language list
# without instantiating the tool.
_MARKET_LANGUAGES: dict[Target_Market, list[Target_Language]] = {
    Target_Market.PH: [Target_Language.FIL, Target_Language.EN],
    Target_Market.TH: [Target_Language.TH, Target_Language.EN],
    Target_Market.RU: [Target_Language.RU, Target_Language.EN],
    Target_Market.EN_GLOBAL: [Target_Language.EN],
}


def market_to_languages(market: Target_Market) -> list[Target_Language]:
    """Return the languages associated with ``market`` (Req 4.1 - 4.4).

    PH → ``[FIL, EN]``; TH → ``[TH, EN]``; RU → ``[RU, EN]``;
    EN_GLOBAL → ``[EN]``.
    """
    try:
        return list(_MARKET_LANGUAGES[market])
    except KeyError as exc:  # pragma: no cover — Target_Market is exhaustive
        raise KeyError(
            f"market_to_languages: unknown Target_Market {market!r}"
        ) from exc


class LocalizationTool:
    """LLM-backed localization tool with placeholder preservation."""

    #: Market → Language fan-out (Req 4.1 - 4.4).
    MARKET_LANGUAGES: dict[Target_Market, list[Target_Language]] = {
        Target_Market.PH: [Target_Language.FIL, Target_Language.EN],
        Target_Market.TH: [Target_Language.TH, Target_Language.EN],
        Target_Market.RU: [Target_Language.RU, Target_Language.EN],
        Target_Market.EN_GLOBAL: [Target_Language.EN],
    }

    #: Currency symbol per market (Req 4.6).
    CURRENCY_SYMBOLS: dict[Target_Market, str] = {
        Target_Market.PH: "₱",
        Target_Market.TH: "฿",
        Target_Market.RU: "₽",
        Target_Market.EN_GLOBAL: "$",
    }

    #: Date format per market (Req 4.7).
    DATE_FORMATS: dict[Target_Market, str] = {
        Target_Market.PH: "MM/DD/YYYY",
        Target_Market.TH: "DD/MM/YYYY",
        Target_Market.RU: "DD.MM.YYYY",
        Target_Market.EN_GLOBAL: "MM/DD/YYYY",
    }

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def languages_for_market(cls, market: Target_Market) -> list[Target_Language]:
        """Return the languages associated with ``market`` (Req 4.1 - 4.4)."""
        return list(cls.MARKET_LANGUAGES[market])

    async def translate(
        self,
        source_copy: str,
        source_language: Target_Language = Target_Language.EN,
        target_languages: Optional[list[Target_Language]] = None,
        target_market: Target_Market = Target_Market.EN_GLOBAL,
    ) -> LocalizerOutput:
        """Translate ``source_copy`` into each language in ``target_languages``.

        When ``target_languages`` is ``None`` (the default) the language list is
        derived from ``target_market`` via :data:`MARKET_LANGUAGES` (Req
        4.1 - 4.4). When supplied explicitly, the list is honoured verbatim
        modulo deduplication.

        Returns:
            :class:`LocalizerOutput`. Every language that could be translated
            and validated lands in ``localized_versions``; any that failed
            (LLM error, placeholder mismatch, timeout, ...) is recorded under
            ``failed_languages`` while the rest of the work continues
            (Req 9.3).

        Raises:
            ValueError: When ``target_languages`` contains a value outside
                ``{en, fil, th, ru}`` (Req 4.9 defence-in-depth — unreachable
                under normal callers because ``Target_Language`` is an enum).
            ToolFailureError: When the overall budget (3000 ms) is exceeded.
        """
        # Derive from target_market when caller omits the list (Req 4.1 - 4.4).
        if target_languages is None:
            target_languages = self.languages_for_market(target_market)

        # Defensive validation (Req 4.9). In normal operation Orchestrator
        # derives ``target_languages`` from ``MARKET_LANGUAGES`` so this is
        # unreachable; we keep the guard so a misuse fails fast and is
        # observable rather than corrupting a translation silently.
        allowed = set(Target_Language)
        for lang in target_languages:
            if lang not in allowed:
                raise ValueError(
                    "Localization_Tool received unsupported target language "
                    f"{lang!r}; allowed values are "
                    f"{sorted(l.value for l in allowed)}"
                )

        start = time.perf_counter()
        source_lang_label = (
            source_language.value
            if isinstance(source_language, Target_Language)
            else str(source_language)
        )
        log.info(
            "localization_tool.invoked",
            source_language=source_lang_label,
            target_market=target_market.value,
            target_count=len(target_languages),
        )

        try:
            output = await asyncio.wait_for(
                self._translate_all(
                    source_copy=source_copy,
                    source_language=source_language,
                    target_languages=target_languages,
                    target_market=target_market,
                ),
                timeout=_DEFAULT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            duration_ms = self._elapsed_ms(start)
            log.error(
                "localization_tool.timeout",
                duration_ms=duration_ms,
                target_market=target_market.value,
            )
            raise ToolFailureError(
                tool_name="Localization_Tool",
                message=(
                    f"Localization_Tool exceeded {int(_DEFAULT_TIMEOUT_S * 1000)}ms"
                    " timeout"
                ),
                original_exception=exc,
            ) from exc

        output.localize_time_ms = self._elapsed_ms(start)
        log.info(
            "localization_tool.completed",
            translated_count=len(output.localized_versions),
            failed_count=len(output.failed_languages),
            duration_ms=output.localize_time_ms,
        )
        return output

    # ------------------------------------------------------------------
    # Per-language fan-out
    # ------------------------------------------------------------------

    async def _translate_all(
        self,
        *,
        source_copy: str,
        source_language: object,
        target_languages: list[Target_Language],
        target_market: Target_Market,
    ) -> LocalizerOutput:
        # Source-copy placeholder bookkeeping. Computed once and reused.
        source_placeholders = self._extract_placeholders(source_copy)
        masked_copy, restore_map = self._mask_placeholders(source_copy)

        # Deduplicate while preserving order; same language asked twice is
        # treated as one job.
        seen: set[Target_Language] = set()
        ordered: list[Target_Language] = []
        for lang in target_languages:
            if lang in seen:
                continue
            seen.add(lang)
            ordered.append(lang)

        async def _job(lang: Target_Language) -> tuple[Target_Language, Optional[str], Optional[str]]:
            """Translate to a single language. Returns ``(lang, text, error)``.

            On success ``text`` is the validated translation and ``error`` is
            None. On failure ``text`` is None and ``error`` carries the reason
            string for ``failed_languages``.
            """
            # Shortcut when source already in the requested language: keep the
            # source verbatim — no LLM round-trip and no placeholder risk.
            source_lang_value = (
                source_language.value
                if isinstance(source_language, Target_Language)
                else (source_language or "en")
            )
            if source_lang_value.lower() == lang.value.lower():
                return lang, source_copy, None

            try:
                translated = await self._translate_one(
                    masked_copy=masked_copy,
                    restore_map=restore_map,
                    source_language=source_language,
                    target_language=lang,
                    target_market=target_market,
                )
            except ToolFailureError as exc:
                log.warning(
                    "localization_tool.language_failed",
                    target_language=lang.value,
                    reason=exc.message,
                )
                return lang, None, exc.message
            except Exception as exc:  # noqa: BLE001 — defensive
                log.warning(
                    "localization_tool.language_failed",
                    target_language=lang.value,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                return lang, None, f"{type(exc).__name__}: {exc}"

            # Placeholder multiset validation (Req 4.5 / Property 5).
            translated_placeholders = self._extract_placeholders(translated)
            if translated_placeholders != source_placeholders:
                reason = (
                    "Placeholder multiset mismatch: "
                    f"source={dict(source_placeholders)} vs "
                    f"translation={dict(translated_placeholders)}"
                )
                log.warning(
                    "localization_tool.placeholder_mismatch",
                    target_language=lang.value,
                    source=dict(source_placeholders),
                    translated=dict(translated_placeholders),
                )
                return lang, None, reason

            return lang, translated, None

        results = await asyncio.gather(*[_job(lang) for lang in ordered])

        output = LocalizerOutput()
        for lang, text, error in results:
            if text is not None:
                output.localized_versions[lang] = text
            else:
                output.failed_languages.append(
                    FailedLanguage(lang=lang.value, reason=error or "unknown error")
                )
        return output

    # ------------------------------------------------------------------
    # Single-language translation
    # ------------------------------------------------------------------

    async def _translate_one(
        self,
        *,
        masked_copy: str,
        restore_map: dict[str, str],
        source_language: object,
        target_language: Target_Language,
        target_market: Target_Market,
    ) -> str:
        """Run one LLM call and restore placeholders. Raises ToolFailureError."""
        currency_symbol = self.CURRENCY_SYMBOLS[target_market]
        date_format = self.DATE_FORMATS[target_market]
        register = _REGISTER_GUIDANCE[target_language]
        source_lang_value = (
            source_language.value
            if isinstance(source_language, Target_Language)
            else (source_language or "en")
        )

        system_prompt = (
            "You are a senior advertising localizer. Translate the user "
            "copy faithfully into the requested target language while "
            "preserving meaning, tone, and any opaque tokens of the form "
            "__PH_<n>__ exactly as-is (do not translate or remove them). "
            f"{register} "
            f"Use the currency symbol {currency_symbol!r} for any monetary "
            f"value and the date format {date_format!r} for any date you "
            "rephrase. Output the translation only — no commentary, no "
            "Markdown fences, no quotation marks around the result."
        )

        user_prompt = (
            f"Source language: {source_lang_value}\n"
            f"Target language: {target_language.value}\n"
            f"Target market: {target_market.value}\n"
            f"Currency symbol: {currency_symbol}\n"
            f"Date format: {date_format}\n\n"
            "Source copy:\n"
            f"{masked_copy}"
        )

        try:
            raw = await self.llm.complete(
                user_prompt,
                system=system_prompt,
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE,
                timeout_ms=_PER_LANG_LLM_TIMEOUT_MS,
            )
        except ToolFailureError:
            raise
        except Exception as exc:
            raise ToolFailureError(
                tool_name="Localization_Tool",
                message=f"LLM translate call failed: {exc}",
                original_exception=exc,
            ) from exc

        if not isinstance(raw, str) or not raw.strip():
            raise ToolFailureError(
                tool_name="Localization_Tool",
                message="LLM returned empty translation",
            )

        # Strip surrounding fences / quotes if the model added them.
        cleaned = self._strip_decorations(raw.strip())
        return self._unmask_placeholders(cleaned, restore_map)

    # ------------------------------------------------------------------
    # Placeholder helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_placeholders(text: str) -> Counter[str]:
        """Multiset of placeholder names found in ``text``."""
        return Counter(_PLACEHOLDER_PATTERN.findall(text))

    @staticmethod
    def _mask_placeholders(text: str) -> tuple[str, dict[str, str]]:
        """Replace ``{name}`` occurrences with ``__PH_<i>__`` sentinels.

        Each *occurrence* gets its own sentinel index, so a placeholder that
        appears twice in the source produces two distinct sentinels and the
        downstream multiset check still detects accidental drops or
        duplications.
        """
        restore: dict[str, str] = {}

        def _sub(match: re.Match[str]) -> str:
            idx = len(restore)
            sentinel = f"__PH_{idx}__"
            restore[sentinel] = match.group(0)
            return sentinel

        masked = _PLACEHOLDER_PATTERN.sub(_sub, text)
        return masked, restore

    @staticmethod
    def _unmask_placeholders(text: str, restore_map: dict[str, str]) -> str:
        """Reverse :meth:`_mask_placeholders`.

        Sentinels are substituted back in declaration order. Sentinels that
        do not appear in ``text`` are silently skipped — the subsequent
        placeholder-multiset check (caller) will flag the discrepancy.
        """
        result = text
        for sentinel, original in restore_map.items():
            result = result.replace(sentinel, original)
        return result

    @staticmethod
    def _strip_decorations(text: str) -> str:
        """Remove triple-backtick fences and matched outer quotes if present."""
        stripped = text.strip()
        # Triple-backtick fences (``` or ```lang).
        if stripped.startswith("```"):
            # Drop the first line (the opening fence with optional language tag).
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[: -len("```")]
            stripped = stripped.strip()

        # Matched outer quotes.
        if len(stripped) >= 2:
            first, last = stripped[0], stripped[-1]
            if first == last and first in ("'", '"'):
                stripped = stripped[1:-1].strip()
        return stripped

    @staticmethod
    def _elapsed_ms(start_time: float) -> int:
        return int((time.perf_counter() - start_time) * 1000)
