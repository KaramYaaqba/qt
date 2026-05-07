"""
Voice Activity Detection

Uses webrtcvad to detect speech/silence boundaries in raw PCM audio.
Accumulates speech frames and fires a callback when a pause is detected.
"""
import collections
import numpy as np

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False


class VADAccumulator:
    """
    Accumulates raw PCM 16kHz mono Int16 frames.
    Fires when PAUSE_MS of silence follows at least MIN_SPEECH_MS of speech.
    Falls back to energy-based VAD if webrtcvad is unavailable.
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 20          # webrtcvad requires 10, 20, or 30ms frames
    PAUSE_MS = 250         # silence after speech → flush utterance
    MIN_SPEECH_MS = 200    # minimum speech before we flush
    AGGRESSIVENESS = 2     # webrtcvad aggressiveness (0-3)
    RESET_PAUSE_MS = 3000  # long silence → reset (user stopped)

    def __init__(self):
        self._frame_bytes = int(self.SAMPLE_RATE * self.FRAME_MS / 1000) * 2  # 2 bytes per Int16
        self._pause_frames = int(self.PAUSE_MS / self.FRAME_MS)
        self._min_speech_frames = int(self.MIN_SPEECH_MS / self.FRAME_MS)
        self._reset_frames = int(self.RESET_PAUSE_MS / self.FRAME_MS)

        if HAS_WEBRTCVAD:
            self._vad = webrtcvad.Vad(self.AGGRESSIVENESS)
        else:
            self._vad = None

        self._speech_buf: list[bytes] = []
        self._silence_count = 0
        self._speech_count = 0
        self._leftover = b""

    def feed(self, pcm_bytes: bytes) -> tuple[np.ndarray | None, bool]:
        """
        Feed raw PCM Int16 bytes.
        Returns (audio, should_reset) where:
          - audio is a float32 array when an utterance is ready, else None
          - should_reset is True when 3+ seconds of silence detected
        """
        data = self._leftover + pcm_bytes
        result = None
        should_reset = False

        i = 0
        while i + self._frame_bytes <= len(data):
            frame = data[i:i + self._frame_bytes]
            i += self._frame_bytes
            is_speech = self._is_speech(frame)

            if is_speech:
                self._speech_buf.append(frame)
                self._speech_count += 1
                self._silence_count = 0
            else:
                if self._speech_count > 0:
                    self._speech_buf.append(frame)
                    self._silence_count += 1

                    if self._silence_count >= self._pause_frames and self._speech_count >= self._min_speech_frames:
                        result = self._flush()
                    elif self._silence_count >= self._reset_frames:
                        self._speech_buf.clear()
                        self._silence_count = 0
                        self._speech_count = 0
                        should_reset = True

        self._leftover = data[i:]
        return result, should_reset

    def _is_speech(self, frame: bytes) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, self.SAMPLE_RATE)
            except Exception:
                pass
        # Energy-based fallback
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        return float(np.abs(samples).mean()) > 200

    def _flush(self) -> np.ndarray:
        raw = b"".join(self._speech_buf)
        self._speech_buf.clear()
        self._silence_count = 0
        self._speech_count = 0
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return samples / 32768.0

    def flush_remaining(self) -> np.ndarray | None:
        """Call on session end to get any remaining accumulated audio."""
        if self._speech_count >= self._min_speech_frames:
            return self._flush()
        self._silence_count = 0
        self._speech_count = 0
        return None
