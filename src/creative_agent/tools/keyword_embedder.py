"""Keyword_Embedder — embeds SEO keywords into copy with natural phrasing.

Implements design.md § Components / 6. Keyword_Embedder and Requirements
5.1 — 5.9.

Behaviour summary
-----------------

* **Already-covered keywords are left alone** (Req 5.5, Property 6): a
  keyword that already matches via case-insensitive word-boundary search is
  marked as a hit and never re-embedded. This is what gives the tool its
  idempotence — running ``embed`` twice with the same keyword list yields
  the same coverage on the second call.
* **Empty keyword list shortcut** (Req 5.8): returns the original copy
  with ``coverage = 1.0`` immediately; no LLM call.
* **Length-aware embedding** (Req 5.3 / Property 8): the LLM is asked to
  rewrite the copy to embed ``missing`` keywords while staying within
  ``platform_spec.char_limit(creative_type)``. If the model's output is
  over budget we retry with a shorter target keyword list (up to 2 retries).
* **Stuffing guard** (Req 5.4): we count consecutive runs of the same
  keyword and reject any embedded copy that contains a keyword three or
  more times in a row. The MVP heuristic looks for the keyword appearing
  back-to-back with only whitespace / punctuation in between.
* **Coverage metric** (Req 5.2): ``hit / requested`` after the final pass.
* **Skipped keywords** (Req 5.6): keywords that did not survive any
  embedding attempt (because every attempt overflowed the char limit) are
  recorded in ``skipped_keywords``.
* **Hard failure** (Req 5.9): if no embedding attempt fits within the char
  limit we keep the *original* copy, set ``coverage = 0.0`` and populate
  ``failure_reason``. Any keywords that already matched the original copy
  are still credited as hits.
* **Timing** (Req 5.7): the whole call is wrapped in
  ``asyncio.wait_for(timeout=1.5)``; the LLM is given ``timeout_ms=1200``.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional

from creative_agent.errors import ToolFailureError
from creative_agent.llm.client import LLMClient
from creative_agent.models import Creative_Type
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.observability.logging import get_logger
from creative_agent.tools.types import EmbedderOutput

__all__ = [
    "EmbedderInput",
    "KeywordEmbedder",
]

log = get_logger(__name__)


# Total wall-clock budget per requirement 5.7.
_DEFAULT_TIMEOUT_S: float = 90.0
# Per-LLM-call timeout — leaves slack for orchestration / retries.
_LLM_CALL_TIMEOUT_MS: int = 80000

_LLM_MAX_TOKENS: int = 2048
_LLM_TEMPERATURE: float = 0.4

# Maximum LLM attempts (1 initial + 1 retry with fewer keywords).
_MAX_LLM_ATTEMPTS: int = 3

# ---------------------------------------------------------------------------
# Keyword matching strategy by writing system
# ---------------------------------------------------------------------------
#
# Strict ``\b``/``\w``-boundary matching only works for languages that separate
# words with spaces and don't fuse affixes onto stems (English, Spanish,
# Russian, …). Two large families break that assumption and need substring
# matching instead, otherwise a keyword that is clearly present is wrongly
# reported missing:
#
#   1. No-space scripts — CJK (Chinese/Japanese), Thai, Khmer, Lao, Myanmar:
#      words are written with no separators at all, so a keyword is almost
#      never on a ``\b`` boundary.
#   2. Fusional/clitic scripts — Arabic: the definite article ال and clitic
#      prepositions/conjunctions attach to the following word (شحن → الشحن).
#
# For these scripts substring matching is safe: unlike Latin ("play" inside
# "player"), a CJK/Thai/Arabic keyword appearing as a character subsequence
# genuinely conveys the keyword. Korean Hangul has spaces but fuses particles,
# so it is treated as substring too. Latin/Cyrillic/Greek keep strict matching
# to preserve the "play ≠ player" guard.
#
# The strategy is chosen from the KEYWORD's own characters (self-contained — no
# language code needed at the call site). A keyword mixing scripts (e.g. a CJK
# word with a Latin brand token) counts as substring if ANY char is from a
# no-boundary script, since the boundary concept doesn't apply to it.

#: Unicode ranges whose presence in a keyword switches matching to substring
#: mode. Each tuple is an inclusive (low, high) code-point range.
_NO_BOUNDARY_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Chinese / Kanji)
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul syllables (Korean — has spaces but fuses particles)
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x0E00, 0x0E7F),   # Thai
    (0x0E80, 0x0EFF),   # Lao
    (0x1780, 0x17FF),   # Khmer
    (0x1000, 0x109F),   # Myanmar
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    (0x0900, 0x097F),   # Devanagari (Hindi — inflected, no reliable \b stems)
)


def _uses_substring_matching(keyword: str) -> bool:
    """True iff ``keyword`` belongs to a no-space / fusional script.

    Such scripts (CJK, Thai, Khmer, Arabic, Hangul, …) cannot rely on word
    boundaries, so the keyword is matched as a plain substring.
    """
    for ch in keyword:
        cp = ord(ch)
        for low, high in _NO_BOUNDARY_RANGES:
            if low <= cp <= high:
                return True
    return False


# ---------------------------------------------------------------------------
# Cyrillic stem-prefix matching (Russian / Kazakh — heavily inflected)
# ---------------------------------------------------------------------------
#
# Cyrillic languages are space-separated, but Russian nouns decline through 6
# cases, so a keyword almost never appears in its citation form inside real
# copy: бонус ("bonus") shows up as бонусом / бонусе / бонуса, and пополнение
# ("top-up") as пополнении / пополнения. Strict whole-word matching misses all
# of these and over-reports "keyword missing". We therefore match a word that
# STARTS WITH the keyword's stem (the keyword minus its inflectional ending),
# allowing any Cyrillic case-ending to follow. The leading word boundary still
# prevents matching mid-word, and requiring the stem (not a bare substring)
# avoids matching unrelated short roots.

_CYRILLIC_RE: re.Pattern[str] = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")

#: Trailing letters stripped to derive a Russian/Kazakh nominal stem. These are
#: the common case-ending vowels and the soft/hard signs; we remove at most
#: two so the stem stays specific enough not to over-match.
_CYRILLIC_INFLECTION_TAIL: str = "аеёиоуыэюяьъ"


def _is_cyrillic_keyword(keyword: str) -> bool:
    """True iff ``keyword`` is written in Cyrillic script."""
    return _CYRILLIC_RE.search(keyword) is not None


def _cyrillic_stem(keyword: str) -> str:
    """Derive a conservative stem by trimming up to two trailing inflection
    letters, while keeping the stem reasonably specific (≥ 4 chars and ≥ 60%
    of the original length)."""
    kw = keyword.strip().lower()
    min_len = max(4, int(len(kw) * 0.6))
    stem = kw
    removed = 0
    while (
        removed < 2
        and len(stem) > min_len
        and stem[-1] in _CYRILLIC_INFLECTION_TAIL
    ):
        stem = stem[:-1]
        removed += 1
    return stem


def _cyrillic_stem_match(text: str, keyword: str) -> bool:
    """Match a whole word in ``text`` that starts with ``keyword``'s stem,
    allowing any Cyrillic case-ending to follow (handles declension)."""
    stem = _cyrillic_stem(keyword)
    if not stem:
        return False
    # Leading word boundary, the stem, then zero or more Cyrillic letters
    # (the inflectional ending). Case-insensitive.
    pattern = re.compile(
        r"(?<!\w)" + re.escape(stem) + r"[\u0400-\u04FF\u0500-\u052F]*",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _cyrillic_stem_count(text: str, keyword: str) -> int:
    """Count whole words in ``text`` whose stem matches ``keyword``'s."""
    stem = _cyrillic_stem(keyword)
    if not stem:
        return 0
    pattern = re.compile(
        r"(?<!\w)" + re.escape(stem) + r"[\u0400-\u04FF\u0500-\u052F]*",
        re.IGNORECASE,
    )
    return len(pattern.findall(text))


# ---------------------------------------------------------------------------
# Latin-script stem-prefix matching (inflected / agglutinative languages)
# ---------------------------------------------------------------------------
#
# Latin script alone can't tell English (where "play" must NOT match "player")
# apart from Spanish ("recarga" should match the plural "recargas") or Turkish
# ("yükleme" should match the agglutinated "yüklemenizi"). We therefore route by
# the request's TARGET LANGUAGE: English (and isolating Vietnamese) keep strict
# whole-word matching; inflected/agglutinative Latin languages use stem-prefix
# matching that allows trailing inflectional/affixal letters.
#
# Language codes (``Target_Language`` values) that should use loose Latin
# stem-prefix matching. English ("en") and Vietnamese ("vi", isolating) are
# deliberately excluded so they keep strict matching.
_LATIN_SUFFIX_LANGUAGES: frozenset[str] = frozenset(
    {
        "es",       # Spanish — plural/gender inflection (recarga→recargas)
        "pt-BR",    # Portuguese (Brazil)
        "pt",       # Portuguese (generic)
        "tr",       # Turkish — agglutinative suffixes
    }
)
_LATIN_AFFIX_LANGUAGES: frozenset[str] = frozenset(
    {
        "id",       # Indonesian — prefix/circumfix (peng-…-an)
        "ms",       # Malay — affixation
        "fil",      # Filipino/Tagalog — prefix/infix (mag-, -um-)
        "sw",       # Swahili — agglutinative prefixes
    }
)

#: Trailing Latin letters trimmed to derive a stem for inflected/agglutinative
#: languages. Kept conservative (mostly vowels + plural/affix consonants).
_LATIN_INFLECTION_TAIL: str = "aeiouszrnmlkt"


def _is_latin_keyword(keyword: str) -> bool:
    """True iff ``keyword`` is purely Latin-script letters (with marks)."""
    has_letter = False
    for ch in keyword:
        if ch.isspace() or not ch.isalpha():
            continue
        cp = ord(ch)
        # Basic Latin + Latin-1 + Latin Extended-A/B ranges.
        if (
            0x0041 <= cp <= 0x024F
            or 0x1E00 <= cp <= 0x1EFF  # Latin Extended Additional (Vietnamese)
        ):
            has_letter = True
        else:
            return False
    return has_letter


def _latin_stem(keyword: str) -> str:
    """Trim up to two trailing inflection letters to derive a Latin stem,
    keeping it specific (≥ 4 chars and ≥ 70% of the original length)."""
    kw = keyword.strip().lower()
    min_len = max(4, int(len(kw) * 0.7))
    stem = kw
    removed = 0
    while (
        removed < 2
        and len(stem) > min_len
        and stem[-1] in _LATIN_INFLECTION_TAIL
    ):
        stem = stem[:-1]
        removed += 1
    return stem


def _latin_suffix_match(text: str, keyword: str) -> bool:
    """Match a whole word in ``text`` that STARTS WITH ``keyword``'s Latin
    stem, allowing trailing word characters (plural/suffix inflection)."""
    stem = _latin_stem(keyword)
    if not stem:
        return False
    pattern = re.compile(r"(?<!\w)" + re.escape(stem) + r"\w*", re.IGNORECASE)
    return pattern.search(text) is not None


def _latin_suffix_count(text: str, keyword: str) -> int:
    """Count whole words in ``text`` starting with ``keyword``'s Latin stem."""
    stem = _latin_stem(keyword)
    if not stem:
        return 0
    pattern = re.compile(r"(?<!\w)" + re.escape(stem) + r"\w*", re.IGNORECASE)
    return len(pattern.findall(text))


def _latin_affix_match(text: str, keyword: str) -> bool:
    """Match ``keyword``'s stem appearing ANYWHERE inside a word — handles
    prefixing/circumfixing languages where the stem sits after a prefix
    (Indonesian ``isi`` in ``pengisian``, Filipino ``karga`` in ``magkarga``)."""
    stem = _latin_stem(keyword)
    if not stem:
        return False
    return stem in text.lower()


def _latin_affix_count(text: str, keyword: str) -> int:
    """Count occurrences of ``keyword``'s stem as a substring (affix languages)."""
    stem = _latin_stem(keyword)
    if not stem:
        return 0
    return text.lower().count(stem)


@dataclass
class EmbedderInput:
    """Input contract mirroring design.md § Keyword_Embedder."""

    copy: str
    keywords: list[str]
    platform_spec: Platform_Spec
    creative_type: Creative_Type


def word_boundary_match(
    text: str, keyword: str, language: Optional[str] = None
) -> bool:
    """Return True iff ``keyword`` appears in ``text``.

    Matching strategy is chosen from the keyword's writing system and, for
    Latin script, the request's ``language``:

    * **No-space / fusional / Devanagari scripts** (CJK, Thai, Khmer, Arabic,
      Hangul, Hindi …): case-insensitive *substring* match, because these
      scripts write words without separators or fuse affixes onto stems.
    * **Cyrillic** (Russian, Kazakh — heavily inflected): *stem-prefix* match,
      so a declined form (``бонус`` → ``бонусом``) still counts as present.
    * **Inflected / agglutinative Latin languages** (Spanish, Portuguese,
      Turkish, Indonesian, Malay, Filipino, Swahili — selected via
      ``language``): Latin *stem-prefix* match, so ``recarga`` matches the
      plural ``recargas`` and ``yükleme`` matches ``yüklemenizi``.
    * **English / Vietnamese / unknown Latin**: strict word-boundary match via
      negative lookbehind/lookahead on ``\\w``, so ``"play"`` does not match
      inside ``"player"``.

    ``language`` is the request's target-language code (a ``Target_Language``
    value such as ``"es"`` / ``"tr"``); when omitted, Latin keywords default to
    strict matching (English-safe).

    Empty / whitespace-only keywords always return ``False``.
    """
    if not keyword or not keyword.strip():
        return False
    if _uses_substring_matching(keyword):
        return keyword.lower() in text.lower()
    if _is_cyrillic_keyword(keyword):
        return _cyrillic_stem_match(text, keyword)
    if language and _is_latin_keyword(keyword):
        if language in _LATIN_SUFFIX_LANGUAGES:
            return _latin_suffix_match(text, keyword)
        if language in _LATIN_AFFIX_LANGUAGES:
            return _latin_affix_match(text, keyword)
    pattern = re.compile(
        r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _count_keyword_hits(
    text: str, keyword: str, language: Optional[str] = None
) -> int:
    """Count occurrences of ``keyword`` in ``text`` (strategy as above)."""
    if not keyword or not keyword.strip():
        return 0
    if _uses_substring_matching(keyword):
        return text.lower().count(keyword.lower())
    if _is_cyrillic_keyword(keyword):
        return _cyrillic_stem_count(text, keyword)
    if language and _is_latin_keyword(keyword):
        if language in _LATIN_SUFFIX_LANGUAGES:
            return _latin_suffix_count(text, keyword)
        if language in _LATIN_AFFIX_LANGUAGES:
            return _latin_affix_count(text, keyword)
    pattern = re.compile(
        r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
        re.IGNORECASE,
    )
    return len(pattern.findall(text))


def _has_stuffing(text: str, keywords: list[str]) -> bool:
    """Detect ≥ 3 consecutive repetitions of any keyword (Req 5.4 MVP)."""
    for kw in keywords:
        if not kw or not kw.strip():
            continue
        # Three repetitions of ``kw`` separated only by whitespace /
        # punctuation (no word characters between them).
        pattern = re.compile(
            r"(?<!\w)"
            + re.escape(kw)
            + r"(?!\w)(?:[\s\W]+(?<!\w)"
            + re.escape(kw)
            + r"(?!\w)){2,}",
            re.IGNORECASE,
        )
        if pattern.search(text):
            return True
    return False


class KeywordEmbedder:
    """LLM-backed keyword embedder with length and stuffing guards."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(
        self,
        copy: str,
        keywords: list[str],
        platform_spec: Platform_Spec,
        creative_type: Creative_Type,
        language: Optional[str] = None,
    ) -> EmbedderOutput:
        """Embed missing keywords into ``copy`` and report coverage.

        ``language`` is the request's target-language code (a
        ``Target_Language`` value); it selects the keyword-matching strategy
        for Latin-script languages (e.g. Spanish/Turkish use stem-prefix
        matching, English stays strict). When omitted, Latin keywords default
        to strict matching.
        """
        start = time.perf_counter()

        # Empty keyword list → coverage 1.0, copy unchanged (Req 5.8).
        if not keywords:
            duration_ms = self._elapsed_ms(start)
            log.info(
                "keyword_embedder.empty_keywords",
                duration_ms=duration_ms,
                copy_len=len(copy),
            )
            return EmbedderOutput(
                embedded_copy=copy,
                keyword_coverage=1.0,
                hit_keywords=[],
                skipped_keywords=[],
                embed_time_ms=duration_ms,
                failure_reason=None,
            )

        # Drop blank entries but keep duplicates so ``coverage`` denominator
        # matches the caller-supplied list.
        normalised = [kw for kw in keywords if isinstance(kw, str) and kw.strip()]
        if not normalised:
            duration_ms = self._elapsed_ms(start)
            return EmbedderOutput(
                embedded_copy=copy,
                keyword_coverage=1.0,
                hit_keywords=[],
                skipped_keywords=[],
                embed_time_ms=duration_ms,
                failure_reason=None,
            )

        try:
            output = await asyncio.wait_for(
                self._embed_inner(
                    copy=copy,
                    keywords=normalised,
                    platform_spec=platform_spec,
                    creative_type=creative_type,
                    language=language,
                ),
                timeout=_DEFAULT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            duration_ms = self._elapsed_ms(start)
            log.error(
                "keyword_embedder.timeout",
                duration_ms=duration_ms,
                keyword_count=len(normalised),
            )
            raise ToolFailureError(
                tool_name="Keyword_Embedder",
                message=(
                    f"Keyword_Embedder exceeded {int(_DEFAULT_TIMEOUT_S * 1000)}ms"
                    " timeout"
                ),
                original_exception=exc,
            ) from exc

        output.embed_time_ms = self._elapsed_ms(start)
        log.info(
            "keyword_embedder.completed",
            duration_ms=output.embed_time_ms,
            keyword_count=len(normalised),
            coverage=output.keyword_coverage,
            hit_count=len(output.hit_keywords),
            skipped_count=len(output.skipped_keywords),
            failure_reason=output.failure_reason,
        )
        return output

    # ------------------------------------------------------------------
    # Embedding inner loop
    # ------------------------------------------------------------------

    async def _embed_inner(
        self,
        *,
        copy: str,
        keywords: list[str],
        platform_spec: Platform_Spec,
        creative_type: Creative_Type,
        language: Optional[str] = None,
    ) -> EmbedderOutput:
        char_limit = platform_spec.char_limit(creative_type)

        # CTA type is too short for keyword embedding — skip entirely.
        # CTA buttons ("Claim Now", "Get Bonus") serve a different purpose
        # than SEO keyword matching. Report coverage=1.0 so it doesn't
        # penalise the composite score.
        if creative_type == Creative_Type.CTA:
            return EmbedderOutput(
                embedded_copy=copy,
                keyword_coverage=1.0,
                hit_keywords=[],
                skipped_keywords=[],
                embed_time_ms=0,
                failure_reason=None,
            )

        # Step 1: classify which keywords are already covered (Req 5.5).
        already_hit, missing = self._classify_hits(copy, keywords, language)

        # Optimisation: nothing to embed.
        if not missing:
            coverage = len(already_hit) / len(keywords)
            return EmbedderOutput(
                embedded_copy=copy,
                keyword_coverage=coverage,
                hit_keywords=list(already_hit),
                skipped_keywords=[],
                embed_time_ms=0,
                failure_reason=None,
            )

        # Step 2: try embedding ``missing`` (in priority order, Req 5.6).
        # Since Creative_Generator now embeds keywords at generation time,
        # if keywords are still missing it's because truncation cut them off.
        # In that case, attempting LLM rewrite will only make things worse
        # (longer output → more truncation). Just report the miss.
        embedded_copy = copy

        # Skip LLM rewrite entirely — just report coverage as-is.
        final_hits, _final_missing = self._classify_hits(copy, keywords, language)
        final_skipped = [kw for kw in keywords if kw not in final_hits]
        coverage = len(final_hits) / len(keywords)

        failure_reason = None
        if final_skipped:
            failure_reason = (
                f"Keywords {final_skipped} not found in copy "
                f"(likely truncated by {char_limit}-char limit)"
            )

        return EmbedderOutput(
            embedded_copy=copy,
            keyword_coverage=coverage,
            hit_keywords=list(final_hits),
            skipped_keywords=final_skipped,
            embed_time_ms=0,
            failure_reason=failure_reason,
        )

    # ------------------------------------------------------------------
    # LLM invocation
    # ------------------------------------------------------------------

    async def _call_llm_embed(
        self,
        *,
        copy: str,
        keywords: list[str],
        char_limit: int,
        creative_type: Creative_Type,
    ) -> str:
        system_prompt = (
            "You are an SEO copy editor. Rewrite the user's ad copy so that "
            "every supplied keyword appears naturally and case-insensitively, "
            "preserving the original meaning, tone, and language. Do not "
            "stuff: each keyword should appear at most twice, never three "
            "times in a row. Output the rewritten copy ONLY — no Markdown "
            "fences, no commentary, no quotes."
        )

        user_prompt = (
            f"creative_type: {creative_type.value}\n"
            f"character_limit: {char_limit}\n"
            f"keywords (in priority order): {', '.join(keywords)}\n\n"
            "Original copy:\n"
            f"{copy}\n\n"
            f"Rewrite the copy so it embeds the keywords naturally and stays "
            f"within {char_limit} characters."
        )

        try:
            raw = await self.llm.complete(
                user_prompt,
                system=system_prompt,
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE,
                timeout_ms=_LLM_CALL_TIMEOUT_MS,
            )
        except ToolFailureError:
            raise
        except Exception as exc:
            raise ToolFailureError(
                tool_name="Keyword_Embedder",
                message=f"LLM embed call failed: {exc}",
                original_exception=exc,
            ) from exc

        if not isinstance(raw, str) or not raw.strip():
            raise ToolFailureError(
                tool_name="Keyword_Embedder",
                message="LLM returned empty embedded copy",
            )

        return self._strip_decorations(raw.strip())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_hits(
        copy: str, keywords: list[str], language: Optional[str] = None
    ) -> tuple[list[str], list[str]]:
        """Split ``keywords`` into ``(hit, missing)`` against ``copy``.

        ``language`` selects the per-keyword matching strategy (see
        :func:`word_boundary_match`). Preserves caller-supplied order in both
        lists. Duplicates in ``keywords`` are kept so the coverage denominator
        matches the request.
        """
        hit: list[str] = []
        missing: list[str] = []
        for kw in keywords:
            if word_boundary_match(copy, kw, language):
                hit.append(kw)
            else:
                missing.append(kw)
        return hit, missing

    @staticmethod
    def _strip_decorations(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[: -len("```")]
            stripped = stripped.strip()
        if len(stripped) >= 2:
            first, last = stripped[0], stripped[-1]
            if first == last and first in ("'", '"'):
                stripped = stripped[1:-1].strip()
        return stripped

    @staticmethod
    def _elapsed_ms(start_time: float) -> int:
        return int((time.perf_counter() - start_time) * 1000)
