"""Forbidden_Term dictionary loader and word-boundary matching helpers.

This module powers the local-dictionary half of the Compliance_Checker
(``design.md`` § 4 *Compliance_Checker*). For each supported language it loads
a JSON file from ``src/creative_agent/config/forbidden_terms/`` and exposes a
case-insensitive word-boundary matcher.

Functions
---------
- :func:`load_forbidden_dictionary` — cached per-language loader.
- :func:`find_term_matches` — case-insensitive, Unicode-aware word-boundary
  matcher returning ``[start, end)`` half-open intervals (per requirement 3.2).

Design notes
------------
The matcher intentionally avoids ``\\b`` from ``re``. ``\\b`` is defined in
terms of the ASCII ``\\w`` class for the standard library's regex engine,
which produces incorrect boundaries when terms or surrounding text contain
non-ASCII letters (Thai, Cyrillic, Filipino accented forms). Instead, we rely
on ``str.isalnum`` — which is Unicode-aware — to verify the characters
immediately before and after each candidate match are *not* alphanumeric (or
that the match touches the string edge). This is the simplified scheme
recommended in the spec for the MVP and is sufficient for the property tests.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from creative_agent.models.enums import (
    Compliance_Severity,
    Target_Language,
    ViolationCategory,
)

_logger = logging.getLogger(__name__)

_FORBIDDEN_TERMS_DIR: Path = Path(__file__).resolve().parent / "forbidden_terms"

# Explicit enum-value → filename mapping (kept independent of enum names so
# enum renames cannot silently break disk lookups).
_LANGUAGE_FILES: dict[Target_Language, str] = {
    Target_Language.EN: "en.json",
    Target_Language.FIL: "fil.json",
    Target_Language.TH: "th.json",
    Target_Language.RU: "ru.json",
}


class ForbiddenTermEntry(BaseModel):
    """A single forbidden / sensitive term entry loaded from a language dictionary.

    Mirrors the ``ForbiddenTermEntry`` interface in ``design.md`` § *Data Models*.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    term: str = Field(min_length=1)
    language: Target_Language
    category: ViolationCategory
    severity: Compliance_Severity
    suggestion: str = Field(min_length=1)


@lru_cache(maxsize=None)
def load_forbidden_dictionary(lang: Target_Language) -> tuple[ForbiddenTermEntry, ...]:
    """Load the forbidden-term dictionary for the given language.

    The cache returns an immutable tuple so callers can safely iterate without
    risking mutation of the cached value (lists are mutable, tuples are not).

    Args:
        lang: Target language whose dictionary should be loaded.

    Returns:
        A tuple of :class:`ForbiddenTermEntry`. If no dictionary is registered
        for the language, or the on-disk file is missing, returns an empty
        tuple after logging a warning. (Per the task spec the function must
        not raise in the missing-file case so the Compliance_Checker can
        gracefully fall back to LLM-only detection for that language.)

    Raises:
        ValueError: If the on-disk JSON exists but fails Pydantic validation
            (e.g. invalid category/severity enum). This is treated as a
            developer-level configuration error.
    """
    filename = _LANGUAGE_FILES.get(lang)
    if filename is None:
        _logger.warning(
            "No forbidden-term dictionary registered for language %r; "
            "returning empty dictionary",
            lang,
        )
        return ()

    config_path = _FORBIDDEN_TERMS_DIR / filename
    if not config_path.is_file():
        _logger.warning(
            "Forbidden-term dictionary file not found for language %r at %s; "
            "returning empty dictionary",
            lang,
            config_path,
        )
        return ()

    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(
            f"Forbidden-term dictionary at {config_path} must be a JSON array"
        )

    entries = tuple(ForbiddenTermEntry.model_validate(item) for item in raw)

    # Defensive: every entry's declared language MUST match the file we
    # loaded it from. Surfacing this here prevents silent cross-language
    # contamination in the Compliance_Checker.
    for entry in entries:
        if entry.language != lang:
            raise ValueError(
                f"Forbidden-term entry {entry.term!r} declares language "
                f"{entry.language!r} but lives in dictionary for {lang!r}"
            )

    return entries


def clear_cache() -> None:
    """Clear the dictionary cache. Primarily useful for tests / hot reload."""
    load_forbidden_dictionary.cache_clear()


def find_term_matches(text: str, term: str) -> list[tuple[int, int]]:
    """Find all case-insensitive, word-boundary-safe matches of ``term`` in ``text``.

    Boundary rule: a candidate match at ``[start, end)`` is accepted iff
    - ``start == 0`` *or* ``text[start - 1]`` is **not** alphanumeric, **and**
    - ``end == len(text)`` *or* ``text[end]`` is **not** alphanumeric.

    "Alphanumeric" is determined via :py:meth:`str.isalnum`, which is
    Unicode-aware in Python 3 and therefore correctly classifies Cyrillic,
    Thai, and Filipino letters/digits.

    Returns half-open intervals (per requirement 3.2 / design § Violation).

    Args:
        text: The text to search. Empty string returns ``[]``.
        term: The term to search for. Empty string returns ``[]``.

    Returns:
        A list of ``(start, end)`` tuples in ascending order.
    """
    if not text or not term:
        return []

    pattern = re.compile(re.escape(term), flags=re.IGNORECASE)

    matches: list[tuple[int, int]] = []
    text_len = len(text)
    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        if start > 0 and text[start - 1].isalnum():
            continue
        if end < text_len and text[end].isalnum():
            continue
        matches.append((start, end))
    return matches


__all__ = [
    "ForbiddenTermEntry",
    "load_forbidden_dictionary",
    "find_term_matches",
    "clear_cache",
]
