"""
Tests for audio processing utilities.
"""
import io
import pytest
import numpy as np
from app.services.audio_processing import (
    _normalize_audio,
    _get_extension,
    validate_audio,
    get_audio_duration,
)


class TestNormalizeAudio:
    def test_float32_output(self):
        audio = np.array([0.5, -0.3, 0.1], dtype=np.float64)
        out = _normalize_audio(audio)
        assert out.dtype == np.float32

    def test_max_is_one(self):
        audio = np.array([2.0, -4.0, 1.0], dtype=np.float32)
        out = _normalize_audio(audio)
        assert abs(np.abs(out).max() - 1.0) < 1e-6

    def test_already_normalized(self):
        audio = np.array([0.5, -1.0, 0.3], dtype=np.float32)
        out = _normalize_audio(audio)
        assert np.allclose(out, audio)

    def test_zero_audio_unchanged(self):
        audio = np.zeros(100, dtype=np.float32)
        out = _normalize_audio(audio)
        assert np.all(out == 0.0)

    def test_sign_preserved(self):
        audio = np.array([-2.0, 1.0], dtype=np.float32)
        out = _normalize_audio(audio)
        assert out[0] < 0
        assert out[1] > 0


class TestGetExtension:
    def test_webm(self):
        assert _get_extension("audio/webm") == ".webm"

    def test_wav_variants(self):
        assert _get_extension("audio/wav") == ".wav"
        assert _get_extension("audio/x-wav") == ".wav"
        assert _get_extension("audio/wave") == ".wav"

    def test_mp3(self):
        assert _get_extension("audio/mpeg") == ".mp3"

    def test_ogg(self):
        assert _get_extension("audio/ogg") == ".ogg"

    def test_unknown_defaults_to_webm(self):
        assert _get_extension("application/octet-stream") == ".webm"

    def test_case_insensitive(self):
        assert _get_extension("Audio/WebM") == ".webm"


class TestValidateAudio:
    def _audio(self, duration_s: float, amplitude: float = 0.5, sr: int = 16000):
        n = int(duration_s * sr)
        return np.full(n, amplitude, dtype=np.float32)

    def test_valid_audio(self):
        audio = self._audio(3.0)
        ok, msg = validate_audio(audio)
        assert ok is True
        assert msg is None

    def test_too_long(self):
        audio = self._audio(35.0)
        ok, msg = validate_audio(audio, max_duration=30.0)
        assert ok is False
        assert "long" in msg.lower()

    def test_too_short(self):
        audio = self._audio(0.3)
        ok, msg = validate_audio(audio, min_duration=0.5)
        assert ok is False
        assert "short" in msg.lower()

    def test_silent_audio(self):
        audio = np.zeros(16000, dtype=np.float32)
        ok, msg = validate_audio(audio)
        assert ok is False
        assert "silent" in msg.lower()

    def test_boundary_exact_max(self):
        audio = self._audio(30.0)
        ok, _ = validate_audio(audio, max_duration=30.0)
        assert ok is True

    def test_boundary_exact_min(self):
        audio = self._audio(0.5)
        ok, _ = validate_audio(audio, min_duration=0.5)
        assert ok is True


class TestGetAudioDuration:
    def test_duration_correct(self):
        audio = np.zeros(32000, dtype=np.float32)
        assert get_audio_duration(audio, 16000) == pytest.approx(2.0)

    def test_one_second(self):
        audio = np.zeros(16000, dtype=np.float32)
        assert get_audio_duration(audio, 16000) == pytest.approx(1.0)
