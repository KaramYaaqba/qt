"""
Audio Processing Service

Handles audio format conversion and preprocessing for speech recognition.
Supports WebM, OGG, MP4, MP3, and WAV formats.
"""
import io
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np


def process_audio(
    audio_bytes: bytes, 
    content_type: str = "audio/webm",
    target_sr: int = 16000
) -> np.ndarray:
    """
    Process audio bytes into a normalized numpy array.
    
    First attempts direct loading via librosa. If that fails (e.g., for WebM),
    falls back to ffmpeg conversion.
    
    Args:
        audio_bytes: Raw audio data
        content_type: MIME type of the audio
        target_sr: Target sample rate (default 16000 for ASR)
        
    Returns:
        Mono audio as numpy float32 array, normalized to [-1, 1]
        
    Raises:
        ValueError: If audio cannot be processed
    """
    import librosa
    
    # Try direct librosa loading first
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        return _normalize_audio(audio)
    except Exception:
        pass
    
    # Fallback: use ffmpeg for format conversion
    ext = _get_extension(content_type)
    
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    out_path = tmp_path + ".wav"
    
    try:
        # Convert to 16kHz mono WAV using ffmpeg
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",                   # Overwrite output
                "-i", tmp_path,         # Input file
                "-ar", str(target_sr),  # Sample rate
                "-ac", "1",             # Mono
                "-f", "wav",            # Output format
                out_path
            ],
            capture_output=True,
            check=True,
            timeout=30
        )
        
        audio, _ = librosa.load(out_path, sr=target_sr, mono=True)
        return _normalize_audio(audio)
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"ffmpeg conversion failed: {e.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise ValueError("ffmpeg conversion timed out")
    except Exception as e:
        raise ValueError(f"Audio processing failed: {str(e)}")
    finally:
        # Cleanup temp files
        Path(tmp_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def _get_extension(content_type: str) -> str:
    """Get file extension for a MIME type."""
    extensions = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/aac": ".aac",
        "audio/x-m4a": ".m4a",
    }
    return extensions.get(content_type.lower(), ".webm")


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio: Input audio array
        
    Returns:
        Normalized audio array
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val

    return audio


def get_audio_duration(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Get duration of audio in seconds.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate


def trim_silence(
    audio: np.ndarray, 
    sample_rate: int = 16000,
    top_db: int = 30
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        top_db: Threshold in dB below reference for silence
        
    Returns:
        Trimmed audio array
    """
    import librosa
    
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def validate_audio(
    audio: np.ndarray, 
    sample_rate: int = 16000,
    max_duration: float = 30.0,
    min_duration: float = 0.5
) -> tuple[bool, Optional[str]]:
    """
    Validate audio meets requirements.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        max_duration: Maximum allowed duration in seconds
        min_duration: Minimum required duration in seconds
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    duration = get_audio_duration(audio, sample_rate)
    
    if duration > max_duration:
        return False, f"Audio too long ({duration:.1f}s > {max_duration}s)"
        
    if duration < min_duration:
        return False, f"Audio too short ({duration:.1f}s < {min_duration}s)"
    
    # Check if audio is too quiet (likely empty)
    if np.abs(audio).max() < 0.01:
        return False, "Audio appears to be silent"
        
    return True, None
