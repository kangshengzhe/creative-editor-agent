"""Tests for Keyword_Embedder's coverage matcher (``word_boundary_match``).

The matcher chooses its strategy from the keyword's writing system:

* No-space / fusional scripts (CJK, Thai, Khmer, Hangul, Arabic, …) use
  case-insensitive *substring* matching, because those scripts write words
  without separators or fuse affixes onto stems, so a keyword that is plainly
  present sits on no ``\\b`` boundary.
* Space-separated, non-fusional scripts (Latin, Cyrillic, …) keep strict
  word-boundary matching so a keyword is never credited as a mere substring of
  an unrelated longer word (e.g. "play" must not match "player").

Regression origin: a Saudi (Arabic) campaign with keyword شحن ("topup") was
wrongly flagged keyword-absent because the copy wrote الشحن (ال "the" + شحن).
Investigation showed the same class of bug affected every no-space script
(Chinese, Thai, Japanese, Khmer) and fusional Hangul, where strict matching
failed almost entirely.
"""

from __future__ import annotations

from creative_agent.tools.keyword_embedder import (
    _count_keyword_hits,
    _uses_substring_matching,
    word_boundary_match,
)


class TestArabicFusedMatching:
    def test_keyword_fused_to_definite_article_matches(self) -> None:
        # الشحن = ال (the) + شحن (recharge). Must be credited as present.
        copy = "وفر عند الشحن: مكافأة ٢٠٪ ترفع"
        assert word_boundary_match(copy, "شحن") is True

    def test_standalone_arabic_keyword_matches(self) -> None:
        assert word_boundary_match("شحن سريع وآمن", "شحن") is True

    def test_absent_arabic_keyword_does_not_match(self) -> None:
        assert word_boundary_match("مكافأة كبيرة اليوم", "شحن") is False

    def test_counts_fused_occurrences(self) -> None:
        assert _count_keyword_hits("الشحن ثم الشحن", "شحن") == 2


class TestNoSpaceScriptsMatchAsSubstring:
    """CJK / Thai / Khmer write words with no separators — substring match."""

    def test_chinese_keyword_inside_run_matches(self) -> None:
        assert word_boundary_match("立即充值享受奖励", "充值") is True

    def test_chinese_absent_keyword_does_not_match(self) -> None:
        assert word_boundary_match("立即享受奖励", "充值") is False

    def test_japanese_katakana_matches(self) -> None:
        assert word_boundary_match("今すぐチャージしてボーナス", "チャージ") is True

    def test_thai_keyword_matches(self) -> None:
        assert word_boundary_match("เติมเงินวันนี้รับโบนัส", "เติมเงิน") is True

    def test_khmer_keyword_matches(self) -> None:
        assert word_boundary_match("បញ្ចូលទឹកប្រាក់ថ្ងៃនេះ", "បញ្ចូលទឹកប្រាក់") is True

    def test_korean_with_fused_particle_matches(self) -> None:
        # 충전 + 하고 (particle) written together.
        assert word_boundary_match("지금 충전하고 보너스", "충전") is True

    def test_chinese_hit_count(self) -> None:
        assert _count_keyword_hits("充值再充值最后充值", "充值") == 3


class TestSpaceSeparatedStaysStrict:
    """Latin / Cyrillic keep strict boundaries (no substring false positives)."""

    def test_english_standalone_matches(self) -> None:
        assert word_boundary_match("press play now", "play") is True

    def test_english_substring_does_not_match(self) -> None:
        # The classic guard: "play" must not match inside "player".
        assert word_boundary_match("the player scored", "play") is False

    def test_english_topup_exact(self) -> None:
        assert word_boundary_match("topup your account", "topup") is True

    def test_english_spaced_variant_does_not_match(self) -> None:
        assert word_boundary_match("top up your account", "topup") is False

    def test_spanish_matches(self) -> None:
        assert word_boundary_match("recarga ahora y gana", "recarga") is True

    def test_russian_matches(self) -> None:
        assert word_boundary_match("пополни счёт сейчас", "пополни") is True

    def test_russian_stem_prefix_is_intentionally_broad(self) -> None:
        # Cyrillic uses stem-prefix matching (see TestCyrillicStemMatching):
        # a 4-char stem like "счёт" therefore also matches the longer
        # "счётчик". This is a deliberate tradeoff — broad matching beats the
        # alternative of missing every declined form (бонус→бонусом etc.).
        assert word_boundary_match("открой счётчик сегодня", "счёт") is True

    def test_empty_keyword_is_false(self) -> None:
        assert word_boundary_match("anything", "") is False
        assert word_boundary_match("anything", "   ") is False


class TestStrategySelection:
    def test_latin_uses_strict(self) -> None:
        assert _uses_substring_matching("topup") is False

    def test_cyrillic_uses_strict(self) -> None:
        assert _uses_substring_matching("пополни") is False

    def test_arabic_uses_substring(self) -> None:
        assert _uses_substring_matching("شحن") is True

    def test_cjk_uses_substring(self) -> None:
        assert _uses_substring_matching("充值") is True

    def test_thai_uses_substring(self) -> None:
        assert _uses_substring_matching("เติมเงิน") is True

    def test_mixed_script_keyword_with_cjk_uses_substring(self) -> None:
        # A CJK keyword carrying a Latin brand token still needs substring mode.
        assert _uses_substring_matching("充值Coco") is True


class TestCyrillicStemMatching:
    """Russian/Kazakh decline nouns through cases — match the stem, not the
    exact citation form, without over-matching unrelated roots."""

    def test_locative_declension_matches(self) -> None:
        # пополнении (locative) of пополнение ("top-up").
        assert word_boundary_match("Бонус при пополнении — без задержек", "пополнение") is True

    def test_instrumental_declension_matches(self) -> None:
        # бонусом (instrumental) of бонус.
        assert word_boundary_match("Пополнение с бонусом — мгновенно", "бонус") is True

    def test_genitive_declension_matches(self) -> None:
        assert word_boundary_match("Получи бонуса сегодня", "бонус") is True

    def test_nominative_exact_matches(self) -> None:
        assert word_boundary_match("Большой бонус ждёт тебя", "бонус") is True

    def test_genuinely_absent_does_not_match(self) -> None:
        assert word_boundary_match("Все больше выбирают пополнение", "бонус") is False

    def test_unrelated_word_does_not_over_match(self) -> None:
        # бонжур ("bonjour") shares leading letters but a different stem.
        assert word_boundary_match("Бонжур, друзья!", "бонус") is False

    def test_stem_counts_multiple_declensions(self) -> None:
        assert _count_keyword_hits("бонус и бонуса и бонусом", "бонус") == 3


class TestLatinLanguageAwareMatching:
    """Latin script alone can't distinguish English (strict) from inflected /
    agglutinative languages; matching is routed by the target ``language``."""

    # --- English / Vietnamese stay strict (no language or en/vi) ----------
    def test_english_strict_without_language(self) -> None:
        assert word_boundary_match("the player scored", "play") is False

    def test_english_strict_with_en(self) -> None:
        assert word_boundary_match("the player scored", "play", "en") is False
        assert word_boundary_match("press play now", "play", "en") is True

    def test_vietnamese_stays_strict(self) -> None:
        assert word_boundary_match("nap tien ngay", "nap", "vi") is True
        assert word_boundary_match("napkin holder", "nap", "vi") is False

    # --- Spanish / Portuguese / Turkish: suffix stem-prefix ---------------
    def test_spanish_plural_matches(self) -> None:
        assert word_boundary_match("recargas instantaneas", "recarga", "es") is True

    def test_spanish_short_plural_matches(self) -> None:
        assert word_boundary_match("bonos exclusivos", "bono", "es") is True

    def test_spanish_genuine_miss(self) -> None:
        assert word_boundary_match("compra ahora", "recarga", "es") is False

    def test_turkish_agglutinated_suffix_matches(self) -> None:
        assert word_boundary_match("yuklemenizi tamamlayin", "yukleme", "tr") is True

    # --- Indonesian / Filipino: affix substring-in-word -------------------
    def test_indonesian_circumfix_matches(self) -> None:
        assert word_boundary_match("pengisian ulang cepat", "isi", "id") is True

    def test_filipino_prefix_matches(self) -> None:
        assert word_boundary_match("magkarga ka ngayon", "karga", "fil") is True

    # --- A Latin keyword without language defaults to strict --------------
    def test_latin_without_language_is_strict(self) -> None:
        # Same Spanish text but no language -> strict, plural NOT matched.
        assert word_boundary_match("recargas instantaneas", "recarga") is False


class TestDevanagariSubstring:
    def test_hindi_keyword_matches_as_substring(self) -> None:
        assert word_boundary_match("अभी रिचार्ज करें और बोनस पाएं", "रिचार्ज") is True

    def test_hindi_absent_keyword_does_not_match(self) -> None:
        assert word_boundary_match("अभी बोनस पाएं", "रिचार्ज") is False
