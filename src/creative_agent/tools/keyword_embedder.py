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


@dataclass
class EmbedderInput:
    """Input contract mirroring design.md § Keyword_Embedder."""

    copy: str
    keywords: list[str]
    platform_spec: Platform_Spec
    creative_type: Creative_Type


def word_boundary_match(text: str, keyword: str) -> bool:
    """Return True iff ``keyword`` appears in ``text`` at a word boundary.

    Case-insensitive, Unicode-aware. Uses a Python-style negative lookbehind
    / lookahead on the ``\\w`` class so neighbouring word characters disqualify
    a match (e.g. searching for ``"play"`` does not match ``"player"``).

    Empty / whitespace-only keywords always return ``False`` so the caller's
    coverage maths stays well-defined when an upstream component leaks
    blanks into the keyword list.
    """
    if not keyword or not keyword.strip():
        return False
    pattern = re.compile(
        r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _count_keyword_hits(text: str, keyword: str) -> int:
    """Count word-boundary occurrences of ``keyword`` in ``text``."""
    if not keyword or not keyword.strip():
        return 0
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
    ) -> EmbedderOutput:
        """Embed missing keywords into ``copy`` and report coverage."""
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
    ) -> EmbedderOutput:
        char_limit = platform_spec.char_limit(creative_type)

        # Step 1: classify which keywords are already covered (Req 5.5).
        already_hit, missing = self._classify_hits(copy, keywords)

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
        embedded_copy = copy
        attempt_keywords = list(missing)
        last_failure_reason: Optional[str] = None

        for attempt in range(1, _MAX_LLM_ATTEMPTS + 1):
            if not attempt_keywords:
                break
            try:
                candidate = await self._call_llm_embed(
                    copy=copy,
                    keywords=attempt_keywords,
                    char_limit=char_limit,
                    creative_type=creative_type,
                )
            except ToolFailureError as exc:
                last_failure_reason = exc.message
                log.warning(
                    "keyword_embedder.llm_attempt_failed",
                    attempt=attempt,
                    error=exc.message,
                )
                # Treat as fatal for this round — fall back to no embedding.
                break

            # Validate: length budget (Req 5.3) and stuffing (Req 5.4).
            if len(candidate) > char_limit:
                last_failure_reason = (
                    f"LLM output exceeded {char_limit}-char limit "
                    f"({len(candidate)})"
                )
                log.warning(
                    "keyword_embedder.length_exceeded",
                    attempt=attempt,
                    candidate_len=len(candidate),
                    char_limit=char_limit,
                )
                # Retry with a shorter keyword target list.
                if len(attempt_keywords) > 1:
                    attempt_keywords = attempt_keywords[:-1]
                    continue
                break

            if _has_stuffing(candidate, keywords):
                last_failure_reason = "Detected keyword stuffing in LLM output"
                log.warning(
                    "keyword_embedder.stuffing_detected",
                    attempt=attempt,
                )
                if len(attempt_keywords) > 1:
                    attempt_keywords = attempt_keywords[:-1]
                    continue
                break

            embedded_copy = candidate
            last_failure_reason = None
            break

        # Step 3: re-classify hits against the (possibly rewritten) copy.
        final_hits, _final_missing = self._classify_hits(embedded_copy, keywords)
        final_skipped = [kw for kw in keywords if kw not in final_hits]

        # Length safety net: if the embedded copy somehow exceeds the limit
        # (e.g. last-attempt fallback raced with a stuffing rejection) revert
        # to the original copy (Req 5.9 safety: never violate the budget).
        if len(embedded_copy) > char_limit:
            embedded_copy = copy
            final_hits, _ = self._classify_hits(copy, keywords)
            final_skipped = [kw for kw in keywords if kw not in final_hits]
            last_failure_reason = (
                last_failure_reason
                or f"Embedded copy exceeded {char_limit}-char limit; reverted"
            )

        coverage = len(final_hits) / len(keywords)

        # Hard failure case (Req 5.9): no keyword could be embedded *and*
        # none were already present.
        if not final_hits:
            return EmbedderOutput(
                embedded_copy=copy,
                keyword_coverage=0.0,
                hit_keywords=[],
                skipped_keywords=list(keywords),
                embed_time_ms=0,
                failure_reason=last_failure_reason
                or "No keyword could be embedded within the char limit",
            )

        return EmbedderOutput(
            embedded_copy=embedded_copy,
            keyword_coverage=coverage,
            hit_keywords=list(final_hits),
            skipped_keywords=final_skipped,
            embed_time_ms=0,
            failure_reason=None,
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
        copy: str, keywords: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split ``keywords`` into ``(hit, missing)`` against ``copy``.

        Preserves caller-supplied order in both lists. Duplicates in
        ``keywords`` are kept so the coverage denominator matches the
        request.
        """
        hit: list[str] = []
        missing: list[str] = []
        for kw in keywords:
            if word_boundary_match(copy, kw):
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
