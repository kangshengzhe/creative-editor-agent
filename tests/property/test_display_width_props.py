"""Property-based tests for Display_Width_Calculator.

Feature: creative-localization-diversity

Hypothesis-driven properties validating the display-unit width behaviour of
``creative_agent.integration.display_width.DisplayWidthCalculator`` against the
design's correctness properties (see design.md § Testing Strategy).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from creative_agent.integration.display_width import DisplayWidthCalculator


# Feature: creative-localization-diversity, Property 12: ASCII display width equals string length
@settings(max_examples=100)
@given(st.text(alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E)))
def test_ascii_display_width_equals_string_length(text: str) -> None:
    """Property 12: ASCII display width equals string length.

    For any text composed solely of ASCII characters (U+0020–U+007E), every
    character is exactly 1 Display_Unit, so the total display width equals the
    Python string length.

    **Validates: Requirements 4.13**
    """
    calculator = DisplayWidthCalculator()
    assert calculator.text_width(text) == len(text)


# Feature: creative-localization-diversity, Property 11: Display width additivity
@settings(max_examples=100)
@given(a=st.text(), b=st.text())
def test_display_width_additivity(a: str, b: str) -> None:
    """Property 11: Display width additivity.

    Concatenating two strings yields a display width equal to the sum of the
    individual display widths, since ``text_width`` is a per-character sum with
    no cross-character interactions.

    **Validates: Requirements 4.14**
    """
    calculator = DisplayWidthCalculator()
    assert calculator.text_width(a + b) == calculator.text_width(a) + calculator.text_width(b)


# Feature: creative-localization-diversity, Property 13: Truncation respects display width limit
@settings(max_examples=100)
@given(text=st.text(), limit=st.integers(min_value=1, max_value=200))
def test_truncation_respects_display_width_limit(text: str, limit: int) -> None:
    """Property 13: Truncation respects display width limit.

    For any text and any positive integer limit, ``truncate_to_width`` produces
    a result whose display width is within the limit, that is a valid
    character-boundary prefix of the original text, and that never splits a
    surrogate pair (Python ``str`` iterates by codepoint, so a character-
    boundary prefix is also a codepoint-boundary prefix).

    **Validates: Requirements 4.9**
    """
    calculator = DisplayWidthCalculator()
    result = calculator.truncate_to_width(text, limit)

    # Result width is within the requested limit.
    assert calculator.text_width(result) <= limit

    # Result is a character-boundary prefix of the original text.
    assert text.startswith(result)
    assert text[: len(result)] == result

    # No surrogate pair is split: a codepoint-boundary prefix re-encodes
    # cleanly to UTF-16 without lone surrogates.
    result.encode("utf-16", "strict")


# ---------------------------------------------------------------------------
# Property 10 strategies
# ---------------------------------------------------------------------------
#
# These strategies generate characters from each classification bucket defined
# by ``display_width.py`` and pair them with their expected display width. They
# are imported directly from the implementation so the test stays in lock-step
# with the documented codepoint ranges.
#
# Precedence note (mirrors ``char_width``): zero-width codepoints and combining
# marks resolve to 0 BEFORE the wide/narrow range checks. So when sampling from
# a wide or narrow range we must filter out any codepoint that is actually a
# combining mark (Unicode category starting with "M") or an explicit zero-width
# codepoint (notably U+FEFF, which lives inside the narrow Arabic Presentation
# Forms-B range U+FE70–U+FEFF). Unassigned codepoints inside a range keep the
# range's width (the implementation only special-cases combining/zero-width),
# so they need no filtering.
import unicodedata as _unicodedata

from creative_agent.integration.display_width import (
    _NARROW_RANGES,
    _WIDE_RANGES,
    _ZERO_WIDTH_CODEPOINTS,
    _ZERO_WIDTH_RANGES,
)


def _codepoints_in(ranges: tuple[tuple[int, int], ...]) -> st.SearchStrategy[int]:
    """Strategy drawing a codepoint uniformly from any of the inclusive ranges."""
    return st.one_of(*[st.integers(min_value=lo, max_value=hi) for lo, hi in ranges])


def _is_zero_width_cp(cp: int) -> bool:
    """True if the codepoint is an explicit zero-width codepoint (Req 4.11)."""
    if cp in _ZERO_WIDTH_CODEPOINTS:
        return True
    return any(lo <= cp <= hi for lo, hi in _ZERO_WIDTH_RANGES)


def _is_combining(char: str) -> bool:
    """True if the character is a combining mark (general category 'M*')."""
    return _unicodedata.category(char).startswith("M")


# Wide chars (expected width 2): CJK ideographs, fullwidth forms, Kana, Hangul.
# Filter out any combining marks so the wide-range assertion holds under the
# implementation's precedence (none are expected, but this keeps it robust).
_wide_chars = (
    _codepoints_in(_WIDE_RANGES)
    .map(chr)
    .filter(lambda c: not _is_combining(c) and not _is_zero_width_cp(ord(c)))
)

# Narrow chars (expected width 1): ASCII/halfwidth, Thai, Arabic. Thai and
# Arabic ranges contain combining marks, and U+FEFF sits inside the Arabic
# Presentation Forms-B range; both resolve to 0 first, so exclude them.
_narrow_chars = (
    _codepoints_in(_NARROW_RANGES)
    .map(chr)
    .filter(lambda c: not _is_combining(c) and not _is_zero_width_cp(ord(c)))
)

# Zero-width chars (expected width 0): explicit ranges + explicit codepoints.
_zero_width_chars = st.one_of(
    _codepoints_in(_ZERO_WIDTH_RANGES),
    st.sampled_from(sorted(_ZERO_WIDTH_CODEPOINTS)),
).map(chr)

# Combining marks (expected width 0): Combining Diacritical Marks block. Filter
# to keep only true category-'M' characters (all are, but stay defensive).
_combining_chars = (
    st.integers(min_value=0x0300, max_value=0x036F).map(chr).filter(_is_combining)
)

# Unclassified / default chars (expected width 1, Req 4.12): IPA Extensions
# (U+0250–U+02AF) fall in none of the defined ranges, are letters (category
# 'Ll'/'Lm'), and are neither combining nor zero-width.
_default_chars = st.integers(min_value=0x0250, max_value=0x02AF).map(chr)

# Tagged union of (char, expected_width) across every classification bucket.
_classified_chars: st.SearchStrategy[tuple[str, int]] = st.one_of(
    _wide_chars.map(lambda c: (c, 2)),
    _narrow_chars.map(lambda c: (c, 1)),
    _zero_width_chars.map(lambda c: (c, 0)),
    _combining_chars.map(lambda c: (c, 0)),
    _default_chars.map(lambda c: (c, 1)),
)


# Feature: creative-localization-diversity, Property 10: Character width classification
@settings(max_examples=100)
@given(case=_classified_chars)
def test_character_width_classification(case: tuple[str, int]) -> None:
    """Property 10: Character width classification.

    For any Unicode character drawn from a defined range, ``char_width`` returns
    the correct display width: 2 for CJK ideographs, fullwidth forms, Kana, and
    Hangul syllables; 1 for ASCII/halfwidth, Thai, and Arabic characters; 0 for
    zero-width characters and combining marks; and 1 for any character not
    otherwise classified. Zero-width/combining detection takes precedence over
    the wide/narrow range checks, so range samples that are combining marks or
    explicit zero-width codepoints are excluded from the wide/narrow buckets.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.11, 4.12**
    """
    char, expected_width = case
    calculator = DisplayWidthCalculator()
    assert calculator.char_width(char) == expected_width
