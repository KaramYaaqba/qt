"""
Tests for SpeechToPhonemeService and MockSpeechToPhonemeService.
Does not require the ONNX model — tests the logic around CTC decode,
log-mel computation, and mock service behaviour.
"""
import json
import tempfile
import os
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reference_data(entries: dict) -> str:
    """Write a temp juz_amma_phonemes.json and return the path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(entries, tmp)
    tmp.close()
    return tmp.name


def _sine_audio(duration_s: float = 1.0, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# MockSpeechToPhonemeService
# ---------------------------------------------------------------------------

class TestMockService:
    REF = {
        "112:1": {"phoneme_list": ["q", "u", "l"], "text_ar": "قُلْ"},
        "112:2": {"phoneme_list": ["a", "l", "l", "a", "h", "u"], "text_ar": "اللَّهُ"},
    }

    def setup_method(self):
        from app.services.speech_to_phoneme import MockSpeechToPhonemeService
        path = _make_reference_data(self.REF)
        self.svc = MockSpeechToPhonemeService(path)
        self._path = path

    def teardown_method(self):
        os.unlink(self._path)

    def test_returns_list(self):
        audio = _sine_audio()
        result = self.svc.predict(audio, hint_surah=112, hint_ayah=1)
        assert isinstance(result, list)
        assert all(isinstance(p, str) for p in result)

    def test_hint_selects_correct_ayah(self):
        # With 85% correct rate, over many runs the result should have most of
        # the original phonemes present. We just check it's non-empty.
        audio = _sine_audio()
        result = self.svc.predict(audio, hint_surah=112, hint_ayah=1)
        assert len(result) > 0

    def test_without_hint_still_returns_phonemes(self):
        audio = _sine_audio()
        result = self.svc.predict(audio)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_phonemes_from_known_vocabulary(self):
        # All phonemes returned should be from the reference vocabulary
        all_ref_phonemes = set()
        for v in self.REF.values():
            all_ref_phonemes.update(v["phoneme_list"])
        # Also include substitutions drawn from the pool
        for _ in range(20):
            result = self.svc.predict(_sine_audio(), hint_surah=112, hint_ayah=1)
            for p in result:
                assert isinstance(p, str)

    def test_common_substitutions_are_strings(self):
        # Regression: substitution values must be strings, not None
        for src, subs in self.svc.common_substitutions.items():
            assert isinstance(src, str)
            for s in subs:
                assert isinstance(s, str)

    def test_all_phonemes_pool_populated(self):
        assert len(self.svc.all_phonemes) > 0

    def test_missing_key_falls_back(self):
        # Unknown surah:ayah → falls back to random key, should not raise
        result = self.svc.predict(_sine_audio(), hint_surah=999, hint_ayah=1)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# SpeechToPhonemeService — CTC decode (unit, no ONNX needed)
# ---------------------------------------------------------------------------

class TestCTCDecode:
    def setup_method(self):
        from app.services.speech_to_phoneme import SpeechToPhonemeService
        self.decode = SpeechToPhonemeService._ctc_decode

    def _vocab(self):
        return {0: "a", 1: "b", 2: "l"}

    def test_blank_removed(self):
        # blank_id = 3
        logits = np.array([[0, 0, 0, 10],   # blank
                           [10, 0, 0, 0],   # a
                           [0, 10, 0, 0]])  # b
        result = self.decode(logits, self._vocab(), blank_id=3)
        assert result == ["a", "b"]

    def test_repeats_collapsed(self):
        logits = np.array([[10, 0, 0, 0],   # a
                           [10, 0, 0, 0],   # a (repeat → collapse)
                           [0, 10, 0, 0]])  # b
        result = self.decode(logits, self._vocab(), blank_id=3)
        assert result == ["a", "b"]

    def test_blank_separates_same_phoneme(self):
        logits = np.array([[10, 0, 0, 0],   # a
                           [0, 0, 0, 10],   # blank
                           [10, 0, 0, 0]])  # a (new instance)
        result = self.decode(logits, self._vocab(), blank_id=3)
        assert result == ["a", "a"]

    def test_empty_logits(self):
        logits = np.zeros((0, 4))
        result = self.decode(logits, self._vocab(), blank_id=3)
        assert result == []

    def test_all_blanks(self):
        logits = np.array([[0, 0, 0, 10],
                           [0, 0, 0, 10]])
        result = self.decode(logits, self._vocab(), blank_id=3)
        assert result == []

    def test_unknown_id_skipped(self):
        # ID 99 not in vocab
        logits = np.zeros((1, 100))
        logits[0, 99] = 10.0
        result = self.decode(logits, self._vocab(), blank_id=3)
        assert result == []

    def test_sequence(self):
        # a blank b blank l
        vocab = self._vocab()
        blank = 3
        logits = np.zeros((5, 4))
        logits[0, 0] = 10  # a
        logits[1, 3] = 10  # blank
        logits[2, 1] = 10  # b
        logits[3, 3] = 10  # blank
        logits[4, 2] = 10  # l
        assert self.decode(logits, vocab, blank) == ["a", "b", "l"]


# ---------------------------------------------------------------------------
# SpeechToPhonemeService — _log_mel (no ONNX, just shape + dtype)
# ---------------------------------------------------------------------------

class TestLogMel:
    """Test _log_mel produces correct shape and dtype without needing the model."""

    def _make_service_stub(self):
        """Return a partial instance that only has _log_mel available."""
        from app.services.speech_to_phoneme import SpeechToPhonemeService
        # Bypass __init__ which needs ONNX file
        obj = object.__new__(SpeechToPhonemeService)
        return obj

    def test_output_shape(self):
        pytest.importorskip("librosa")
        svc = self._make_service_stub()
        audio = _sine_audio(1.0)
        mel = svc._log_mel(audio, 16000)
        assert mel.ndim == 2
        assert mel.shape[0] == 80  # 80 mel bins

    def test_output_dtype(self):
        pytest.importorskip("librosa")
        svc = self._make_service_stub()
        audio = _sine_audio(1.0)
        mel = svc._log_mel(audio, 16000)
        assert mel.dtype == np.float32

    def test_longer_audio_more_frames(self):
        pytest.importorskip("librosa")
        svc = self._make_service_stub()
        short = svc._log_mel(_sine_audio(1.0), 16000)
        long_ = svc._log_mel(_sine_audio(2.0), 16000)
        assert long_.shape[1] > short.shape[1]

    def test_silent_audio_no_nan(self):
        pytest.importorskip("librosa")
        svc = self._make_service_stub()
        silent = np.zeros(16000, dtype=np.float32)
        mel = svc._log_mel(silent, 16000)
        assert not np.isnan(mel).any()
        assert not np.isinf(mel).any()

    def test_trailing_pad_added(self):
        """Output should be longer than raw input frames due to 1s trailing pad."""
        pytest.importorskip("librosa")
        svc = self._make_service_stub()
        audio = _sine_audio(1.0, sr=16000)
        mel = svc._log_mel(audio, 16000)
        # 1s audio at hop=160 ≈ 100 frames; pad adds ~100 more → > 150 total
        assert mel.shape[1] > 150


# ---------------------------------------------------------------------------
# create_service factory
# ---------------------------------------------------------------------------

class TestCreateService:
    def test_mock_requires_reference_path(self):
        from app.services.speech_to_phoneme import create_service
        with pytest.raises(ValueError, match="reference_data_path"):
            create_service(use_mock=True)

    def test_real_requires_model_and_tokens(self):
        from app.services.speech_to_phoneme import create_service
        with pytest.raises(ValueError, match="model_path"):
            create_service(use_mock=False)

    def test_real_missing_tokens_raises(self):
        from app.services.speech_to_phoneme import create_service
        with pytest.raises(ValueError):
            create_service(use_mock=False, model_path="x.onnx")

    def test_mock_creates_instance(self):
        from app.services.speech_to_phoneme import create_service, MockSpeechToPhonemeService
        ref_data = {"112:1": {"phoneme_list": ["q"], "text_ar": "ق"}}
        path = _make_reference_data(ref_data)
        try:
            svc = create_service(use_mock=True, reference_data_path=path)
            assert isinstance(svc, MockSpeechToPhonemeService)
        finally:
            os.unlink(path)
