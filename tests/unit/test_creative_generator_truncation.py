"""Tests for word-boundary-aware truncation in CreativeGenerator._truncate.

Real ads never ship a copy that ends in a half word ("...20% more value, s" or
"...Your Topup Bo"). For space-separated scripts the truncator must retreat to
the last whole word that fits and drop dangling separator punctuation, staying
at or under the character limit. CJK / no-space scripts keep Display_Unit
character truncation (each glyph is a complete unit, so no partial-word issue).
"""

from __future__ import annotations

from creative_agent.tools.creative_generator import CreativeGenerator

_t = CreativeGenerator._truncate


class TestWordBoundaryTruncation:
    def test_does_not_split_trailing_word(self) -> None:
        out = _t("topup bonus: 20% more value, save now", 30)
        assert len(out) <= 30
        assert out == "topup bonus: 20% more value"
        # No partial trailing word, no dangling comma.
        assert not out.endswith(",")

    def test_retreats_to_whole_word(self) -> None:
        out = _t("24/7 Support for Your Topup Bonus today", 30)
        assert len(out) <= 30
        assert out == "24/7 Support for Your Topup"

    def test_strips_dangling_dash(self) -> None:
        out = _t("20% more value per dollar—topup bonus", 30)
        assert len(out) <= 30
        # The em dash joins "dollar—topup" into one token (no surrounding
        # spaces), so the partial "dollar—topu" is dropped entirely.
        assert not out.endswith("—")
        assert out == "20% more value per"

    def test_short_text_unchanged(self) -> None:
        text = "Fast topup bonus now"
        assert _t(text, 30) == text

    def test_strips_separator_left_at_end_after_retreat(self) -> None:
        # After retreating, the kept text ends on a separator token that should
        # be trimmed: "Save big - massive ..." cut so "massive" is partial.
        out = _t("Save big - massive instant bonus today", 16)
        assert len(out) <= 16
        assert not out.endswith("-")
        assert out == "Save big"

    def test_exact_fit_unchanged(self) -> None:
        text = "x" * 30
        assert _t(text, 30) == text

    def test_single_long_word_hard_slice_fallback(self) -> None:
        # No space to retreat to → return a clipped single word, never empty.
        out = _t("Supercalifragilisticexpialidocious", 10)
        assert out == "Supercalif"
        assert len(out) == 10

    def test_never_exceeds_limit(self) -> None:
        samples = [
            "Get 20% more game credits on topup now",
            "Reliable instant delivery for every gamer worldwide",
            "Trust topup bonus: 20% extra, instant credit",
        ]
        for s in samples:
            for limit in (10, 20, 25, 30, 40):
                out = _t(s, limit)
                assert len(out) <= limit, (s, limit, out)
                # Result is whitespace-trimmed.
                assert out == out.strip()

    def test_zero_limit_returns_empty(self) -> None:
        assert _t("anything", 0) == ""


class TestCJKTruncationUnaffected:
    """No-space CJK copy keeps Display_Unit character truncation."""

    def test_cjk_truncates_by_display_units(self) -> None:
        # Each CJK char is width 2; limit 10 display units => 5 chars.
        out = _t("春限定大特価セール", 10, use_display_width=True)
        # Result is whole CJK characters within the display-width budget.
        assert out  # non-empty
        # Every character kept is a full CJK glyph (no partial-unit splitting).
        assert all(ord(ch) > 0x2E7F for ch in out)
