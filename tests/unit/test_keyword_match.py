"""Tests for word-boundary keyword matching."""
import pytest
from creative_agent.config.forbidden_loader import find_term_matches


class TestFindTermMatches:
    def test_basic_match(self):
        matches = find_term_matches("hello world", "world")
        assert matches == [(6, 11)]

    def test_case_insensitive(self):
        matches = find_term_matches("Hello WORLD", "world")
        assert matches == [(6, 11)]

    def test_no_match(self):
        matches = find_term_matches("hello world", "xyz")
        assert matches == []

    def test_word_boundary_rejects_substring(self):
        # "play" should NOT match inside "player"
        matches = find_term_matches("the player scored", "play")
        assert matches == []

    def test_word_boundary_accepts_standalone(self):
        matches = find_term_matches("press play now", "play")
        assert matches == [(6, 10)]

    def test_multiple_occurrences(self):
        matches = find_term_matches("bet big, bet small", "bet")
        assert len(matches) == 2
        assert matches[0] == (0, 3)
        assert matches[1] == (9, 12)

    def test_at_start_of_string(self):
        matches = find_term_matches("guaranteed win", "guaranteed")
        assert matches == [(0, 10)]

    def test_at_end_of_string(self):
        matches = find_term_matches("this is guaranteed", "guaranteed")
        assert matches == [(8, 18)]

    def test_empty_text(self):
        assert find_term_matches("", "word") == []

    def test_empty_term(self):
        assert find_term_matches("hello", "") == []

    def test_unicode_cyrillic(self):
        # Russian term matching
        matches = find_term_matches("это гарантировано работает", "гарантировано")
        assert len(matches) == 1

    def test_unicode_thai(self):
        matches = find_term_matches("ปลอดภัย 100% สำหรับทุกคน", "ปลอดภัย 100%")
        assert len(matches) == 1

    def test_punctuation_boundary(self):
        # Comma after term should count as boundary
        matches = find_term_matches("guaranteed, this works", "guaranteed")
        assert matches == [(0, 10)]
