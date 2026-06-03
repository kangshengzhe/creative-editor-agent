"""Display_Width_Calculator — display-unit width of text strings.

Implements design.md § Components / 1. Display_Width_Calculator and
Requirements 4.1 — 4.14.

The calculator is a stateless pure function operating on Unicode codepoints.
Ad platforms (notably Google Ads) measure copy length in *Display_Units*:
CJK and other "wide" characters occupy 2 units, ASCII/halfwidth occupy 1,
and zero-width / combining marks occupy 0. ``len()`` is therefore the wrong
tool for CJK markets; this module is the single source of truth for
converting text to Display_Unit counts.

Classification (per ``char_width``)
-----------------------------------

Width **2** (wide):
    * CJK ideographs: U+4E00–U+9FFF, U+3400–U+4DBF, U+F900–U+FAFF,
      U+20000–U+2A6DF, U+2A700–U+2B73F, U+2B740–U+2B81F, U+2B820–U+2CEAF
      (Req 4.1)
    * Fullwidth forms: U+FF01–U+FF60, U+FFE0–U+FFE6 (Req 4.2)
    * Japanese Kana: U+3040–U+309F (Hiragana), U+30A0–U+30FF (Katakana)
      (Req 4.4)
    * Korean Hangul syllables: U+AC00–U+D7AF (Req 4.5)

Width **1** (narrow):
    * Halfwidth / ASCII: U+0020–U+007E, U+FF61–U+FFDC (Req 4.3)
    * Thai: U+0E00–U+0E7F (Req 4.6)
    * Arabic: U+0600–U+06FF, U+0750–U+077F, U+FB50–U+FDFF, U+FE70–U+FEFF
      (Req 4.7)

Width **0** (zero-width):
    * Zero-width characters: U+200B–U+200F, U+FEFF, U+2060 (Req 4.11)
    * Combining marks: Unicode general category starting with ``M`` (Req 4.11)

Width **1** otherwise (default fallback, Req 4.12).

Note on precedence: zero-width and combining-mark detection take priority so
that, e.g., U+FEFF (which falls inside no wide range) is correctly 0 and
combining marks are 0 regardless of their codepoint.
"""

from __future__ import annotations

import unicodedata

__all__ = ["DisplayWidthCalculator"]


# --- Wide (2 Display_Units) codepoint ranges -------------------------------
# Each tuple is an inclusive (low, high) range of Unicode codepoints.
_WIDE_RANGES: tuple[tuple[int, int], ...] = (
    # CJK ideographs (Req 4.1)
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
    # Fullwidth forms (Req 4.2)
    (0xFF01, 0xFF60),
    (0xFFE0, 0xFFE6),
    # Japanese Kana (Req 4.4)
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    # Korean Hangul syllables (Req 4.5)
    (0xAC00, 0xD7AF),
)

# --- Narrow (1 Display_Unit) codepoint ranges ------------------------------
_NARROW_RANGES: tuple[tuple[int, int], ...] = (
    # Halfwidth / ASCII (Req 4.3)
    (0x0020, 0x007E),
    (0xFF61, 0xFFDC),
    # Thai (Req 4.6)
    (0x0E00, 0x0E7F),
    # Arabic (Req 4.7)
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)

# --- Zero-width (0 Display_Units) explicit codepoints (Req 4.11) -----------
_ZERO_WIDTH_RANGES: tuple[tuple[int, int], ...] = (
    (0x200B, 0x200F),  # ZWSP, ZWNJ, ZWJ, LRM, RLM
)
_ZERO_WIDTH_CODEPOINTS: frozenset[int] = frozenset({0xFEFF, 0x2060})


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    """Return True if *codepoint* falls within any inclusive range."""
    for low, high in ranges:
        if low <= codepoint <= high:
            return True
    return False


class DisplayWidthCalculator:
    """Stateless calculator for display-unit width of text strings."""

    def char_width(self, char: str) -> int:
        """Return the display width (0, 1, or 2) of a single character.

        Width 0 for zero-width characters and combining marks; 2 for CJK
        ideographs, fullwidth forms, Kana, and Hangul syllables; 1 for
        ASCII/halfwidth, Thai, Arabic, and any otherwise-unclassified
        character (default fallback per Req 4.12).
        """
        codepoint = ord(char)

        # Zero-width: explicit codepoints take precedence over range checks
        # so U+FEFF (BOM) and U+2060 (word joiner) are correctly 0 (Req 4.11).
        if (
            codepoint in _ZERO_WIDTH_CODEPOINTS
            or _in_ranges(codepoint, _ZERO_WIDTH_RANGES)
        ):
            return 0

        # Combining marks → Unicode general category beginning with "M"
        # (Mn, Mc, Me). Width 0 (Req 4.11).
        if unicodedata.category(char).startswith("M"):
            return 0

        # Wide characters → 2 Display_Units (Req 4.1, 4.2, 4.4, 4.5).
        if _in_ranges(codepoint, _WIDE_RANGES):
            return 2

        # Explicit narrow ranges → 1 Display_Unit (Req 4.3, 4.6, 4.7).
        if _in_ranges(codepoint, _NARROW_RANGES):
            return 1

        # Default fallback (Req 4.12).
        return 1

    def text_width(self, text: str) -> int:
        """Return the total display width of a string in Display_Units.

        Equal to the sum of each character's :meth:`char_width`. An empty
        string (or a string of only zero-width characters) yields 0.
        """
        return sum(self.char_width(ch) for ch in text)

    def truncate_to_width(self, text: str, max_width: int) -> str:
        """Truncate text from the end until total width <= ``max_width``.

        Removes whole characters (Python ``str`` iterates by codepoint, so a
        surrogate pair stored as a single astral codepoint is one unit and is
        never split). The result is always a character-boundary prefix of the
        original text. A non-positive ``max_width`` yields an empty string.
        """
        if max_width <= 0:
            return ""

        total = self.text_width(text)
        if total <= max_width:
            return text

        # Walk from the end, dropping one character at a time until the
        # running width fits. Operating on the prefix length keeps the result
        # a valid character-boundary prefix.
        end = len(text)
        while end > 0 and total > max_width:
            end -= 1
            total -= self.char_width(text[end])
        return text[:end]

    def fits_within_limit(self, text: str, limit: int) -> bool:
        """Check if text fits within the given display-unit limit."""
        return self.text_width(text) <= limit
