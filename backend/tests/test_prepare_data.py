"""
Tests for prepare_data.py — data pipeline utilities.
Does not download any datasets; tests pure functions only.
"""
import pytest
import sys
from pathlib import Path

# Make training module importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))

from prepare_data import (
    normalize,
    assign_split,
    extract_vocabulary,
    process_ea_ud,
    _DEV_MOD,
    _TEST_MOD,
    TARGET_SURAHS,
    SURAH_NAME_TO_NUMBER,
)


class TestNormalize:
    def test_strips_diacritics(self):
        # fatha (U+064E) should be removed
        text = "بَسْمِ"
        result = normalize(text)
        assert "َ" not in result  # fatha
        assert "ْ" not in result  # sukun

    def test_normalizes_alef_variants(self):
        # آ, أ, إ, ٱ all → ا
        assert normalize("آمن") == "امن"
        assert normalize("أحد") == "احد"
        assert normalize("إبراهيم") == "ابراهيم"
        assert normalize("ٱللَّهُ") == "الله"  # also strips diacritics

    def test_collapses_whitespace(self):
        assert normalize("  كلمة   أخرى  ") == "كلمة اخرى"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_plain_arabic_unchanged(self):
        assert normalize("الله") == "الله"


class TestAssignSplit:
    def test_test_split_on_test_mod(self):
        assert assign_split(_TEST_MOD) == "test"

    def test_dev_split_on_dev_mod(self):
        assert assign_split(_DEV_MOD) == "dev"

    def test_train_for_other_indices(self):
        for i in range(10):
            if i not in (_TEST_MOD, _DEV_MOD):
                assert assign_split(i) == "train"

    def test_deterministic(self):
        for i in range(100):
            assert assign_split(i) == assign_split(i)

    def test_modular_repeats(self):
        # idx 0 and idx 10 should have the same split
        assert assign_split(0) == assign_split(10)
        assert assign_split(1) == assign_split(11)


class TestExtractVocabulary:
    def test_blank_appended_last(self):
        manifests = {"train": [{"text": "a b c"}]}
        vocab = extract_vocabulary(manifests)
        assert vocab[-1] == "<blank>"

    def test_all_tokens_present(self):
        manifests = {"train": [{"text": "a b"}, {"text": "b c"}]}
        vocab = extract_vocabulary(manifests)
        assert "a" in vocab
        assert "b" in vocab
        assert "c" in vocab

    def test_no_duplicates(self):
        manifests = {"train": [{"text": "a a b b"}]}
        vocab = extract_vocabulary(manifests)
        tokens_no_blank = [t for t in vocab if t != "<blank>"]
        assert len(tokens_no_blank) == len(set(tokens_no_blank))

    def test_sorted_except_blank(self):
        manifests = {"train": [{"text": "z a m b"}]}
        vocab = extract_vocabulary(manifests)
        tokens_no_blank = [t for t in vocab if t != "<blank>"]
        assert tokens_no_blank == sorted(tokens_no_blank)

    def test_empty_manifests(self):
        vocab = extract_vocabulary({"train": []})
        assert vocab == ["<blank>"]

    def test_multi_split(self):
        manifests = {
            "train": [{"text": "a b"}],
            "dev": [{"text": "c d"}],
            "test": [{"text": "e"}],
        }
        vocab = extract_vocabulary(manifests)
        for t in ["a", "b", "c", "d", "e"]:
            assert t in vocab


class TestProcessEaUd:
    """Unit tests for process_ea_ud — no network calls, uses fake data."""

    def _sample(self, text="بِسْمِ اللَّهِ", duration=5.0):
        return {"transcription": text, "duration": duration, "audio": {"bytes": b""}}

    def test_too_long_skipped(self):
        sample = self._sample(duration=31.0)
        entry, reason = process_ea_ud(sample, 0, None, None, {})
        assert entry is None
        assert reason == "too_long"

    def test_no_match_skipped(self):
        sample = self._sample(text="لا يوجد تطابق")
        _, reason = process_ea_ud(sample, 0, None, None, {})
        assert reason == "no_match"

    def test_boundary_30s_not_skipped(self):
        # 30.0s exactly should not be filtered by too_long (> 30, not >=)
        sample = self._sample(duration=30.0)
        # Will fail on no_match since lookup is empty — that's fine
        _, reason = process_ea_ud(sample, 0, None, None, {})
        assert reason != "too_long"

    def test_no_audio_bytes_skipped(self):
        # Matched in lookup but audio decode will fail on empty bytes
        lookup = {normalize("بسم الله"): (1, 1)}
        sample = {"transcription": "بِسْمِ اللَّهِ", "duration": 5.0, "audio": {"bytes": b""}}
        entry, reason = process_ea_ud(sample, 0, None, None, lookup)
        assert entry is None


class TestTargetSurahs:
    def test_range_starts_at_67(self):
        assert min(TARGET_SURAHS) == 67

    def test_range_ends_at_114(self):
        assert max(TARGET_SURAHS) == 114

    def test_count(self):
        assert len(TARGET_SURAHS) == 48  # 67–114 inclusive

    def test_surah_name_map_coverage(self):
        # Every value in the name map should be in TARGET_SURAHS
        for name, number in SURAH_NAME_TO_NUMBER.items():
            assert number in TARGET_SURAHS, f"{name} ({number}) not in TARGET_SURAHS"

    def test_al_ikhlas_in_map(self):
        assert "Al-Ikhlas" in SURAH_NAME_TO_NUMBER
        assert SURAH_NAME_TO_NUMBER["Al-Ikhlas"] == 112

    def test_an_nas_in_map(self):
        assert "An-Nas" in SURAH_NAME_TO_NUMBER
        assert SURAH_NAME_TO_NUMBER["An-Nas"] == 114
