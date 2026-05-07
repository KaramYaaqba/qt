"""
Speech-to-Phoneme Service

Two implementations:
1. SpeechToPhonemeService - Real ONNX model inference for Conformer-CTC
2. MockSpeechToPhonemeService - Returns corrupted reference phonemes for testing
"""
import logging
import numpy as np
import json
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Logit-domain stride merge parameters (NeMo buffered inference algorithm)
_BUFFER_S = 8.0      # total window fed to model per chunk (seconds)
_CHUNK_S = 4.0       # center portion kept per chunk (seconds)
_STRIDE_S = (_BUFFER_S - _CHUNK_S) / 2   # 2s context on each side
_HOP = 160           # mel hop_length (samples)
_MODEL_STRIDE = 8    # FastConformer subsampling factor
_SR = 16000
# One encoder frame = _HOP * _MODEL_STRIDE samples = 0.08s
_FRAME_S = _HOP * _MODEL_STRIDE / _SR   # 0.08s per encoder frame
_SHORT_S = 6.0  # audio shorter than this → single pass


class SpeechToPhonemeService:
    """Real ONNX model inference for Conformer-CTC."""

    def __init__(self, model_path: str, tokens_path: str):
        import onnxruntime as ort
        
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )

        # Load vocabulary from tokens file
        self.vocab = {}
        with open(tokens_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.vocab[int(parts[1])] = parts[0]
        
        # CTC blank token is typically the last one
        self.blank_id = max(self.vocab.keys()) if self.vocab else 0

        # Cache input names for inference
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def _log_mel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute log-mel spectrogram matching NeMo's AudioToMelSpectrogramPreprocessor.

        Explicit settings from training config:
          preemph=0.97, mel_norm="slaney", log_zero_guard=2**-24, mag_power=2.0
        Dither is training-only (self.training gate in NeMo) — not applied here.
        """
        import librosa

        # 1. Preemphasis (always applied, not training-only)
        audio = np.concatenate([[audio[0]], audio[1:] - 0.97 * audio[:-1]])

        # 2. STFT → power spectrum (mag_power=2.0)
        stft = librosa.stft(
            audio, n_fft=512, hop_length=_HOP,
            win_length=400, window='hann', center=True,
        )
        power = np.abs(stft) ** 2  # mag_power=2.0

        # 3. Mel filterbank with Slaney norm (matches NeMo mel_norm="slaney")
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=512,
            n_mels=80, fmin=0, fmax=8000,
            norm='slaney',
        )
        mel = mel_basis @ power  # (80, T)

        # 4. Log with NeMo guard value (2**-24, log_zero_guard_type="add")
        lm = np.log(mel + 2**-24)

        # 5. Per-feature normalization (normalize="per_feature")
        lm = (lm - lm.mean(axis=1, keepdims=True)) / (lm.std(axis=1, keepdims=True) + 1e-9)

        # 6. Trailing pad so final phonemes are decoded cleanly
        pad = np.zeros((lm.shape[0], int(1.0 / 0.01)), dtype=np.float32)
        return np.concatenate([lm, pad], axis=1)

    def _run_onnx(self, log_mel: np.ndarray) -> np.ndarray:
        """Run ONNX session → raw logits (T, vocab_size)."""
        inp = log_mel[np.newaxis, :, :].astype(np.float32)
        length = np.array([log_mel.shape[1]], dtype=np.int64)
        return self.session.run(
            None, {self.input_names[0]: inp, self.input_names[1]: length}
        )[0][0]

    @staticmethod
    def _ctc_decode(logits: np.ndarray, vocab: dict, blank_id: int) -> list[str]:
        """Greedy CTC decode: collapse repeats, remove blanks."""
        ids = np.argmax(logits, axis=-1)
        phonemes, prev = [], -1
        for idx in ids:
            if idx != prev and idx != blank_id and idx in vocab:
                phonemes.append(vocab[idx])
            prev = idx
        return phonemes

    def predict(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        hint_surah: Optional[int] = None,
        hint_ayah: Optional[int] = None,
    ) -> list[str]:
        """
        Convert audio to phoneme sequence.

        Short audio (≤ 9s): single pass.
        Long audio: logit-domain stride merge (NeMo buffered inference algorithm).
          - Split into overlapping windows of _BUFFER_S seconds
          - Keep only the center _CHUNK_S seconds of each window's logits
          - Concatenate kept logits and decode once
          This avoids token-boundary artifacts from naive sliding window.
        """
        duration = len(audio_array) / sample_rate
        logger.info(f"Audio duration: {duration:.2f}s")

        if duration <= _SHORT_S:
            logits = self._run_onnx(self._log_mel(audio_array, sample_rate))
            phonemes = self._ctc_decode(logits, self.vocab, self.blank_id)
            logger.info(f"Predicted {len(phonemes)} phonemes (single pass): {phonemes[:8]}")
            return phonemes

        # --- Logit-domain stride merge ---
        buf_samples = int(_BUFFER_S * sample_rate)
        step_samples = int(_CHUNK_S * sample_rate)
        stride_samples = int(_STRIDE_S * sample_rate)
        keep_frames = int(_CHUNK_S / _FRAME_S)
        drop_frames = int(_STRIDE_S / _FRAME_S)

        # Pad so the last chunk is complete
        audio_padded = np.pad(audio_array, (stride_samples, buf_samples), mode='constant')

        all_logits = []
        start = 0
        n_chunks = 0
        while start + buf_samples <= len(audio_padded):
            chunk = audio_padded[start: start + buf_samples]
            logits = self._run_onnx(self._log_mel(chunk, sample_rate))
            # Drop left-stride frames, keep center _CHUNK_S worth of frames
            center = logits[drop_frames: drop_frames + keep_frames]
            if center.shape[0] > 0:
                all_logits.append(center)
            start += step_samples
            n_chunks += 1

        if not all_logits:
            return []

        merged = np.concatenate(all_logits, axis=0)
        phonemes = self._ctc_decode(merged, self.vocab, self.blank_id)
        logger.info(
            f"Predicted {len(phonemes)} phonemes ({n_chunks} chunks, "
            f"logit-merge): {phonemes[:8]}"
        )
        return phonemes


class MockSpeechToPhonemeService:
    """
    Returns slightly corrupted reference phonemes for testing.
    Use this while the real model is being trained on RunPod.
    
    This simulates what a trained model would do by taking the
    correct phoneme sequence and introducing realistic errors.
    """

    def __init__(self, reference_data_path: str):
        """
        Load reference phoneme data.
        
        Args:
            reference_data_path: Path to juz_amma_phonemes.json
        """
        with open(reference_data_path, encoding="utf-8") as f:
            self.references = json.load(f)
        
        # Build set of all phonemes for realistic substitutions/insertions
        self.all_phonemes = set()
        for v in self.references.values():
            self.all_phonemes.update(v.get("phoneme_list", []))
        self.all_phonemes = list(self.all_phonemes)
        
        # Common substitution pairs for more realistic errors
        self.common_substitutions = {
            "sˤ": ["s", "z"],      # emphatic s
            "dˤ": ["d", "t"],      # emphatic d
            "tˤ": ["t", "d"],      # emphatic t
            "ðˤ": ["ð", "z"],      # emphatic dh
            "ħ": ["h", "x"],       # pharyngeal h
            "ʕ": ["ʔ", "a"],       # pharyngeal ayn
            "ʁ": ["r", "ɣ"],       # uvular r (ghain)
            "q": ["k", "ɡ"],       # uvular q
            "θ": ["s", "t"],       # th
            "ð": ["d", "z"],       # dh
        }

    def predict(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int = 16000,
        hint_surah: Optional[int] = None,
        hint_ayah: Optional[int] = None
    ) -> list[str]:
        """
        Return phonemes with simulated errors.
        
        Args:
            audio_array: Audio samples (unused, just for API compatibility)
            sample_rate: Sample rate (unused)
            hint_surah: If provided, use this surah's reference
            hint_ayah: If provided, use this ayah's reference
            
        Returns:
            List of phoneme symbols with ~15% simulated errors
        """
        # If we know which ayah the user selected, use that reference
        if hint_surah is not None and hint_ayah is not None:
            key = f"{hint_surah}:{hint_ayah}"
        else:
            # Random ayah for testing
            key = random.choice(list(self.references.keys()))

        phonemes = list(self.references.get(key, {}).get("phoneme_list", ["a"]))

        # Introduce ~15% random errors to simulate a real model
        result = []
        for p in phonemes:
            r = random.random()
            if r < 0.10:
                # Substitution (10% chance)
                # Try to use realistic substitution if available
                if p in self.common_substitutions and random.random() < 0.7:
                    result.append(random.choice(self.common_substitutions[p]))
                else:
                    result.append(random.choice(self.all_phonemes))
            elif r < 0.13:
                # Deletion (3% chance) — skip this phoneme
                continue
            elif r < 0.15:
                # Insertion (2% chance) — add an extra phoneme
                result.append(random.choice(self.all_phonemes))
                result.append(p)
            else:
                # Correct (85% chance)
                result.append(p)
                
        return result


def create_service(
    use_mock: bool = True,
    model_path: Optional[str] = None,
    tokens_path: Optional[str] = None,
    reference_data_path: Optional[str] = None
):
    """
    Factory function to create appropriate service based on configuration.
    
    Args:
        use_mock: If True, use MockSpeechToPhonemeService
        model_path: Path to ONNX model (required if use_mock=False)
        tokens_path: Path to tokens.txt (required if use_mock=False)
        reference_data_path: Path to reference data (required if use_mock=True)
        
    Returns:
        Either MockSpeechToPhonemeService or SpeechToPhonemeService
    """
    if use_mock:
        if reference_data_path is None:
            raise ValueError("reference_data_path required for mock service")
        return MockSpeechToPhonemeService(reference_data_path)
    else:
        if model_path is None or tokens_path is None:
            raise ValueError("model_path and tokens_path required for real service")
        return SpeechToPhonemeService(model_path, tokens_path)
