"""
Speech-to-Phoneme Service

Two implementations:
1. SpeechToPhonemeService - Real ONNX model inference for Conformer-CTC
2. MockSpeechToPhonemeService - Returns corrupted reference phonemes for testing
"""
import numpy as np
import json
import random
from pathlib import Path
from typing import Optional


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

    def predict(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int = 16000,
        hint_surah: Optional[int] = None,
        hint_ayah: Optional[int] = None
    ) -> list[str]:
        """
        Convert audio to phoneme sequence using ONNX model.
        
        Args:
            audio_array: Audio samples as numpy array (mono, 16kHz)
            sample_rate: Sample rate (should be 16000)
            hint_surah: Unused in real model, kept for API compatibility
            hint_ayah: Unused in real model, kept for API compatibility
            
        Returns:
            List of phoneme symbols
        """
        import librosa

        # Compute 80-dim log-mel spectrogram matching NeMo defaults
        mel = librosa.feature.melspectrogram(
            y=audio_array,
            sr=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
            fmin=0,
            fmax=8000
        )
        
        # Convert to log scale
        log_mel = np.log(mel + 1e-9).T  # Shape: (time_steps, 80)
        
        # Normalize per feature
        log_mel = (log_mel - log_mel.mean(axis=0)) / (log_mel.std(axis=0) + 1e-9)

        # Prepare inputs for ONNX model
        input_signal = log_mel[np.newaxis, :, :].astype(np.float32)
        input_length = np.array([log_mel.shape[0]], dtype=np.int64)

        # Run inference
        outputs = self.session.run(
            None,
            {
                self.input_names[0]: input_signal,
                self.input_names[1]: input_length,
            }
        )

        # Get logits and decode
        logits = outputs[0][0]  # Shape: (time_steps, vocab_size)
        predicted_ids = np.argmax(logits, axis=-1)

        # CTC greedy decode: collapse repeats, remove blanks
        phonemes = []
        prev_id = -1
        for idx in predicted_ids:
            if idx != prev_id and idx != self.blank_id:
                if idx in self.vocab:
                    phonemes.append(self.vocab[idx])
            prev_id = idx
            
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
