"""
Tests for Pydantic schemas — validates field presence, constraints, and
that new Tajweed fields serialize correctly.
"""
import pytest
from pydantic import ValidationError
from app.models.schemas import (
    LetterResult,
    PhonemeError,
    RecitationCheckRequest,
    RecitationCheckResponse,
)


class TestLetterResult:
    def _base(self, **kwargs):
        defaults = {"letter": "ب", "status": "correct", "position": 0}
        defaults.update(kwargs)
        return defaults

    def test_minimal_valid(self):
        lr = LetterResult(**self._base())
        assert lr.letter == "ب"
        assert lr.status == "correct"

    def test_tajweed_fields_optional(self):
        lr = LetterResult(**self._base())
        assert lr.tajweed_status is None
        assert lr.tajweed_error_type is None
        assert lr.expected_phoneme_full is None
        assert lr.got_phoneme_full is None

    def test_tajweed_fields_populated(self):
        lr = LetterResult(**self._base(
            tajweed_status="error",
            tajweed_error_type="replace",
            expected_phoneme_full="sˤ",
            got_phoneme_full="s",
        ))
        assert lr.tajweed_status == "error"
        assert lr.tajweed_error_type == "replace"
        assert lr.expected_phoneme_full == "sˤ"
        assert lr.got_phoneme_full == "s"

    def test_all_optional_fields_none_by_default(self):
        lr = LetterResult(**self._base())
        assert lr.error_type is None
        assert lr.expected_phoneme is None
        assert lr.got_phoneme is None

    def test_diacritic_status(self):
        lr = LetterResult(letter="َ", status="diacritic", position=1)
        assert lr.status == "diacritic"

    def test_serialization_includes_tajweed_fields(self):
        lr = LetterResult(**self._base(tajweed_status="correct"))
        d = lr.model_dump()
        assert "tajweed_status" in d
        assert "tajweed_error_type" in d
        assert "expected_phoneme_full" in d
        assert "got_phoneme_full" in d


class TestPhonemeError:
    def test_valid(self):
        e = PhonemeError(
            type="replace",
            position_in_expected=2,
            position_in_predicted=2,
            expected_phoneme="q",
            got_phoneme="k",
        )
        assert e.type == "replace"

    def test_optional_phonemes_none(self):
        e = PhonemeError(
            type="insert",
            position_in_expected=0,
            position_in_predicted=0,
        )
        assert e.expected_phoneme is None
        assert e.got_phoneme is None


class TestRecitationCheckRequest:
    def test_valid_surah_67(self):
        # surah 67 should now be valid (schema was fixed from ge=78 to ge=67)
        req = RecitationCheckRequest(surah=67, ayah=1)
        assert req.surah == 67

    def test_valid_surah_114(self):
        req = RecitationCheckRequest(surah=114, ayah=3)
        assert req.surah == 114

    def test_surah_66_invalid(self):
        with pytest.raises(ValidationError):
            RecitationCheckRequest(surah=66, ayah=1)

    def test_surah_115_invalid(self):
        with pytest.raises(ValidationError):
            RecitationCheckRequest(surah=115, ayah=1)

    def test_ayah_0_invalid(self):
        with pytest.raises(ValidationError):
            RecitationCheckRequest(surah=112, ayah=0)

    def test_ayah_1_valid(self):
        req = RecitationCheckRequest(surah=112, ayah=1)
        assert req.ayah == 1


class TestRecitationCheckResponse:
    def _make(self, **kwargs):
        defaults = dict(
            surah=112, ayah=1,
            reference_text="قُلْ هُوَ اللَّهُ أَحَدٌ",
            accuracy_phoneme=85.7,
            accuracy_letter=88.9,
            total_phonemes=14,
            total_errors=2,
            phoneme_errors=[],
            letter_results=[],
        )
        defaults.update(kwargs)
        return RecitationCheckResponse(**defaults)

    def test_valid(self):
        r = self._make()
        assert r.surah == 112

    def test_letter_results_with_tajweed_fields(self):
        lr = LetterResult(
            letter="ص",
            status="correct",
            position=0,
            tajweed_status="error",
            tajweed_error_type="replace",
            expected_phoneme_full="sˤ",
            got_phoneme_full="s",
        )
        r = self._make(letter_results=[lr])
        assert r.letter_results[0].tajweed_status == "error"

    def test_serialization_roundtrip(self):
        r = self._make()
        d = r.model_dump()
        r2 = RecitationCheckResponse(**d)
        assert r2.accuracy_phoneme == r.accuracy_phoneme
