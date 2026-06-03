"""Example-based unit tests for Display_Width_Calculator edge cases.

Feature: creative-localization-diversity (task 2.6)

Covers the display-width edge cases enumerated in design.md
§ Error Handling / "Display Width Edge Cases":

- Empty string -> width 0
- String of only zero-width characters -> width 0
- Surrogate pair / astral character at a truncation boundary is never split
  (the whole codepoint is dropped rather than partially included)
- Unknown / uncategorized character defaults to width 1

Validates: Requirements 4.11, 4.12
"""

from __future__ import annotations

import pytest

from creative_agent.integration.display_width import DisplayWidthCalculator

# --- Sample characters -----------------------------------------------------
# Astral (supplementary-plane) CJK Extension B ideograph: a single Python
# codepoint (len == 1) that occupies 2 Display_Units. Internally it is stored
# as a UTF-16 surrogate pair, so truncation must never split it.
ASTRAL_CJK = chr(0x20000)  # 𠀀, category Lo, width 2

# Private-Use-Area codepoint (category Co): falls into no wide/narrow/
# zero-width range and is not a combining mark, so it hits the default
# fallback width of 1 (Req 4.12).
PRIVATE_USE = chr(0xE000)  # category Co, width 1 (default fallback)

# Zero-width characters (Req 4.11).
ZERO_WIDTH_SPACE = "\u200b"  # U+200B ZERO WIDTH SPACE
BYTE_ORDER_MARK = "\ufeff"  # U+FEFF ZERO WIDTH NO-BREAK SPACE / BOM


class TestTextWidthEdgeCases:
    """Edge cases for ``text_width``."""

    def test_empty_string_has_zero_width(self) -> None:
        """An empty string has a display width of 0 (Req 4.14 base case)."""
        calculator = DisplayWidthCalculator()
        assert calculator.text_width("") == 0

    def test_zero_width_only_string_has_zero_width(self) -> None:
        """A string of only zero-width characters has display width 0 (Req 4.11)."""
        calculator = DisplayWidthCalculator()
        zero_width_only = ZERO_WIDTH_SPACE + BYTE_ORDER_MARK + "\u200d" + "\u2060"
        assert calculator.text_width(zero_width_only) == 0

    def test_astral_character_is_width_two_single_codepoint(self) -> None:
        """An astral CJK ideograph is one Python codepoint of width 2 (Req 4.1)."""
        calculator = DisplayWidthCalculator()
        assert len(ASTRAL_CJK) == 1
        assert calculator.char_width(ASTRAL_CJK) == 2
        assert calculator.text_width(ASTRAL_CJK) == 2


class TestUnknownCharacterDefault:
    """Default fallback width for uncategorized characters (Req 4.12)."""

    def test_private_use_char_defaults_to_width_one(self) -> None:
        """A Private-Use-Area codepoint is not classified and defaults to 1."""
        calculator = DisplayWidthCalculator()
        assert calculator.char_width(PRIVATE_USE) == 1

    def test_unknown_char_within_string_contributes_width_one(self) -> None:
        """The default-width char sums normally within a mixed string."""
        calculator = DisplayWidthCalculator()
        # "A" (1) + PUA (1) + astral CJK (2) = 4
        assert calculator.text_width("A" + PRIVATE_USE + ASTRAL_CJK) == 4


class TestTruncationNeverSplitsAstralCharacter:
    """``truncate_to_width`` drops whole astral codepoints, never splitting."""

    def test_astral_char_dropped_whole_when_it_cannot_fit(self) -> None:
        """A width-2 astral char is dropped entirely when the limit is 1.

        The calculator must not include a partial (1-unit) slice of the
        2-unit character; the only character-boundary prefix that fits is the
        empty string.
        """
        calculator = DisplayWidthCalculator()
        result = calculator.truncate_to_width(ASTRAL_CJK, 1)
        assert result == ""
        # The surrogate pair was never split: the prefix re-encodes cleanly.
        result.encode("utf-16", "strict")

    def test_astral_char_at_boundary_dropped_as_whole_unit(self) -> None:
        """At a boundary, the whole astral char is dropped, not half of it."""
        calculator = DisplayWidthCalculator()
        # "A" (1) + astral (2) = total width 3; limit 2 forces dropping the
        # astral char entirely, leaving the width-1 prefix "A".
        text = "A" + ASTRAL_CJK
        result = calculator.truncate_to_width(text, 2)
        assert result == "A"
        assert calculator.text_width(result) <= 2
        result.encode("utf-16", "strict")

    def test_astral_char_retained_when_it_fits(self) -> None:
        """When the limit accommodates the full width-2 char, it is kept whole."""
        calculator = DisplayWidthCalculator()
        text = "A" + ASTRAL_CJK
        result = calculator.truncate_to_width(text, 3)
        assert result == text
        result.encode("utf-16", "strict")

    def test_zero_width_chars_never_force_truncation(self) -> None:
        """Zero-width chars add no width, so a zero-width-only string is kept."""
        calculator = DisplayWidthCalculator()
        zero_width_only = ZERO_WIDTH_SPACE + BYTE_ORDER_MARK
        result = calculator.truncate_to_width(zero_width_only, 1)
        assert result == zero_width_only


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
