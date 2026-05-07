"""
Tests for AlignmentService — covers both passes, normalization, letter mapping,
beginner mode, tajweed mode, edge cases, and accuracy calculation.
"""
import pytest
from app.services.alignment import (
    AlignmentService,
    _normalize_phoneme,
    _PHONEME_NORM,
    _BEGINNER_NORM,
    _PHONEME_MULTI,
    BEGINNER_MODE,
)


# ---------------------------------------------------------------------------
# _normalize_phoneme
# ---------------------------------------------------------------------------

class TestNormalizePhoneme:
    def test_unicode_colon_normalized(self):
        # ː (modifier colon) → : (ascii colon)
        assert _normalize_phoneme("aː", beginner=False) == "a:"

    def test_script_g_normalized(self):
        assert _normalize_phoneme("ɡ", beginner=False) == "g"

    def test_nasalized_n_normalized(self):
        assert _normalize_phoneme("ñ", beginner=False) == "n"

    def test_velar_nasal_normalized(self):
        assert _normalize_phoneme("ŋ", beginner=False) == "n"

    def test_nasalized_m_multi(self):
        assert _normalize_phoneme("m̃", beginner=False) == "m"

    def test_qalqala_stripped(self):
        assert _normalize_phoneme("Q", beginner=False) == ""

    def test_plain_phoneme_unchanged(self):
        assert _normalize_phoneme("b", beginner=False) == "b"

    def test_emphatic_s_unchanged_non_beginner(self):
        assert _normalize_phoneme("sˤ", beginner=False) == "sˤ"

    # Beginner mode collapses
    def test_emphatic_s_collapsed_beginner(self):
        assert _normalize_phoneme("sˤ", beginner=True) == "s"

    def test_emphatic_vowel_collapsed_beginner(self):
        assert _normalize_phoneme("aˤ", beginner=True) == "a"

    def test_long_vowel_collapsed_beginner(self):
        assert _normalize_phoneme("a:", beginner=True) == "a"
        assert _normalize_phoneme("i:", beginner=True) == "i"
        assert _normalize_phoneme("u:", beginner=True) == "u"

    def test_geminate_collapsed_beginner(self):
        assert _normalize_phoneme("bb", beginner=True) == "b"
        assert _normalize_phoneme("nn", beginner=True) == "n"

    def test_hamza_empty_in_beginner(self):
        assert _normalize_phoneme("ʔ", beginner=True) == ""

    def test_hamza_kept_non_beginner(self):
        assert _normalize_phoneme("ʔ", beginner=False) == "ʔ"

    def test_raa_heavy_collapsed_beginner(self):
        assert _normalize_phoneme("rˤ", beginner=True) == "r"


# ---------------------------------------------------------------------------
# AlignmentService — perfect match
# ---------------------------------------------------------------------------

class TestAlignPerfect:
    def setup_method(self):
        self.svc = AlignmentService()

    def test_perfect_match_accuracy(self):
        phonemes = ["q", "u", "l"]
        result = self.svc.align(phonemes, phonemes, "قُلْ")
        assert result["accuracy_phoneme"] == 100.0
        assert result["total_errors"] == 0
        assert result["phoneme_errors"] == []

    def test_perfect_match_no_letter_errors(self):
        phonemes = ["b", "a"]
        result = self.svc.align(phonemes, phonemes, "بَ")
        error_letters = [lr for lr in result["letter_results"] if lr.get("status") == "error"]
        assert error_letters == []

    def test_empty_expected_returns_empty_result(self):
        result = self.svc.align(["a", "b"], [], "test")
        assert result["accuracy_phoneme"] == 0.0
        assert result["letter_results"] != []  # letters still produced


# ---------------------------------------------------------------------------
# AlignmentService — error detection
# ---------------------------------------------------------------------------

class TestAlignErrors:
    def setup_method(self):
        self.svc = AlignmentService()

    def test_substitution_detected(self):
        expected = ["q", "u", "l"]
        predicted = ["k", "u", "l"]  # q → k
        result = self.svc.align(predicted, expected, "قُلْ")
        assert result["total_errors"] >= 1
        types = [e["type"] for e in result["phoneme_errors"]]
        assert "replace" in types

    def test_deletion_detected(self):
        expected = ["q", "u", "l"]
        predicted = ["q", "l"]  # u dropped
        result = self.svc.align(predicted, expected, "قُلْ")
        types = [e["type"] for e in result["phoneme_errors"]]
        assert "insert" in types or "delete" in types

    def test_insertion_detected(self):
        expected = ["q", "u"]
        predicted = ["q", "a", "u"]  # extra 'a'
        result = self.svc.align(predicted, expected, "قُ")
        assert result["total_errors"] >= 1

    def test_all_wrong_low_accuracy(self):
        expected = ["q", "u", "l"]
        predicted = ["x", "i", "m"]
        result = self.svc.align(predicted, expected, "قُلْ")
        assert result["accuracy_phoneme"] < 50.0

    def test_accuracy_phoneme_range(self):
        expected = ["a", "b", "a", "b"]
        predicted = ["a", "x", "a", "b"]  # 1 error
        result = self.svc.align(predicted, expected, "test")
        assert 0.0 <= result["accuracy_phoneme"] <= 100.0


# ---------------------------------------------------------------------------
# AlignmentService — beginner mode: hamza skipped
# ---------------------------------------------------------------------------

class TestBeginnerModeHamza:
    def setup_method(self):
        self.svc = AlignmentService()

    def test_hamza_not_penalised_in_beginner(self):
        # ʔ normalises to "" in beginner mode — missing it should not count as error
        expected = ["ʔ", "a", "l", "l", "a", "h"]
        predicted = ["a", "l", "l", "a", "h"]  # no hamza
        result = self.svc.align(predicted, expected, "اللَّهُ")
        # Beginner pass strips ʔ from expected before alignment,
        # so predicted vs filtered-expected should match perfectly
        assert result["accuracy_phoneme"] == 100.0

    def test_hamza_present_also_fine(self):
        expected = ["ʔ", "a"]
        predicted = ["ʔ", "a"]
        result = self.svc.align(predicted, expected, "أَ")
        assert result["accuracy_phoneme"] == 100.0


# ---------------------------------------------------------------------------
# AlignmentService — tajweed pass: emphatics caught
# ---------------------------------------------------------------------------

class TestTajweedPass:
    def setup_method(self):
        self.svc = AlignmentService()

    def test_emphatic_correct_in_beginner_error_in_tajweed(self):
        # sˤ said as s — beginner mode passes (both → s), tajweed mode catches it
        expected = ["sˤ", "a"]
        predicted = ["s", "a"]
        result = self.svc.align(predicted, expected, "صَ")
        # Beginner accuracy should be 100 (both normalize to s/a)
        assert result["accuracy_phoneme"] == 100.0
        # Tajweed should flag an error on letter with sˤ
        letter_tajweed_errors = [
            lr for lr in result["letter_results"]
            if lr.get("tajweed_status") == "error"
        ]
        assert len(letter_tajweed_errors) >= 1

    def test_long_vowel_correct_in_beginner_error_in_tajweed(self):
        expected = ["a:"]
        predicted = ["a"]
        result = self.svc.align(predicted, expected, "آ")
        # beginner collapses both to a — no error
        assert result["accuracy_phoneme"] == 100.0
        # tajweed detects a: vs a
        letter_tajweed_errors = [
            lr for lr in result["letter_results"]
            if lr.get("tajweed_status") == "error"
        ]
        assert len(letter_tajweed_errors) >= 1


# ---------------------------------------------------------------------------
# AlignmentService — letter result fields
# ---------------------------------------------------------------------------

class TestLetterResultFields:
    def setup_method(self):
        self.svc = AlignmentService()

    def test_correct_letter_has_correct_status(self):
        result = self.svc.align(["b", "a"], ["b", "a"], "بَ")
        letters = [lr for lr in result["letter_results"] if lr.get("status") == "correct"]
        assert len(letters) >= 1

    def test_error_letter_has_error_type(self):
        result = self.svc.align(["k"], ["q"], "قُ")
        errors = [lr for lr in result["letter_results"] if lr.get("status") == "error"]
        if errors:
            assert errors[0]["error_type"] in ("replace", "insert", "delete")

    def test_diacritic_status(self):
        # Arabic fatha diacritic (U+064E) should be tagged as diacritic
        result = self.svc.align(["b", "a"], ["b", "a"], "بَ")  # بَ = ب + fatha
        diacritics = [lr for lr in result["letter_results"] if lr.get("status") == "diacritic"]
        assert len(diacritics) >= 1

    def test_tajweed_status_field_present_on_letters(self):
        result = self.svc.align(["b", "a"], ["b", "a"], "بَ")
        letters = [lr for lr in result["letter_results"] if lr.get("status") == "correct"]
        for lr in letters:
            assert "tajweed_status" in lr

    def test_expected_phoneme_full_present(self):
        result = self.svc.align(["sˤ"], ["sˤ"], "صَ")
        letters = [lr for lr in result["letter_results"] if lr.get("status") in ("correct", "error")]
        if letters:
            assert "expected_phoneme_full" in letters[0]


# ---------------------------------------------------------------------------
# AlignmentService — letter_phoneme_map path
# ---------------------------------------------------------------------------

class TestLetterPhonemeMap:
    def setup_method(self):
        self.svc = AlignmentService()

    def _make_map(self, chars_list, phoneme_counts):
        """Build a minimal letter_phoneme_map."""
        entries = []
        offset = 0
        for chars, count in zip(chars_list, phoneme_counts):
            entries.append({"chars": chars, "start": offset, "end": offset + count})
            offset += count
        return entries

    def test_map_path_correct(self):
        lpm = self._make_map(["بَ", "ب"], [1, 1])
        result = self.svc.align(["b", "b"], ["b", "b"], "بَب", letter_phoneme_map=lpm)
        assert result["accuracy_phoneme"] == 100.0

    def test_map_path_error_propagates_to_letter(self):
        lpm = self._make_map(["قَ"], [1])
        result = self.svc.align(["k"], ["q"], "قَ", letter_phoneme_map=lpm)
        letters = [lr for lr in result["letter_results"] if lr.get("status") == "error"]
        assert len(letters) >= 1

    def test_map_path_and_sequential_agree_on_accuracy(self):
        expected = ["b", "a", "b"]
        predicted = ["b", "x", "b"]
        lpm = self._make_map(["ب", "َ", "ب"], [1, 1, 1])
        r_map = self.svc.align(predicted, expected, "بَب", letter_phoneme_map=lpm)
        r_seq = self.svc.align(predicted, expected, "بَب")
        # Both paths should compute the same phoneme accuracy
        assert r_map["accuracy_phoneme"] == r_seq["accuracy_phoneme"]


# ---------------------------------------------------------------------------
# AlignmentService — accuracy_letter
# ---------------------------------------------------------------------------

class TestAccuracyLetter:
    def setup_method(self):
        self.svc = AlignmentService()

    def test_letter_accuracy_100_when_perfect(self):
        result = self.svc.align(["b", "a"], ["b", "a"], "بَ")
        assert result["accuracy_letter"] == 100.0

    def test_letter_accuracy_in_range(self):
        result = self.svc.align(["k"], ["q"], "ق")
        assert 0.0 <= result["accuracy_letter"] <= 100.0

    def test_diacritics_not_counted_in_letter_accuracy(self):
        # Only consonant 'ب' is scorable; diacritic 'َ' should not count
        result_with = self.svc.align(["b", "a"], ["b", "a"], "بَ")
        result_without = self.svc.align(["b"], ["b"], "ب")
        # Both should have 100% letter accuracy — diacritic doesn't affect it
        assert result_with["accuracy_letter"] == 100.0
        assert result_without["accuracy_letter"] == 100.0


# ---------------------------------------------------------------------------
# AlignmentService — _empty_result
# ---------------------------------------------------------------------------

class TestEmptyResult:
    def setup_method(self):
        self.svc = AlignmentService()

    def test_empty_expected_gives_zero_accuracy(self):
        result = self.svc.align(["a", "b"], [], "بَ")
        assert result["accuracy_phoneme"] == 0.0
        assert result["accuracy_letter"] == 0.0

    def test_empty_expected_letter_results_still_populated(self):
        result = self.svc.align([], [], "بَ")
        assert len(result["letter_results"]) > 0

    def test_empty_everything(self):
        result = self.svc.align([], [], "")
        assert result["accuracy_phoneme"] == 0.0
