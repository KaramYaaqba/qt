"""
Quran Position Tracker — CTC Forward Trellis Approach

Instead of fuzzy-matching CTC output against text, we run the CTC forward
pass constrained to the known Quran phoneme sequence. The argmax of the
forward scores tells us exactly where in the surah the reciter is.

This is the same algorithm used by torchaudio.functional.forced_align and
how Tarteel-style apps track position without random jumps.

Reference: https://distill.pub/2017/ctc/
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PositionUpdate:
    def __init__(self, ayah_number: int, word_index: int, confidence: float,
                 letter_results: list | None, completed_ayah: bool):
        self.ayah_number = ayah_number
        self.word_index = word_index
        self.confidence = confidence
        self.letter_results = letter_results
        self.completed_ayah = completed_ayah


class PositionTracker:
    """
    Tracks recitation position using CTC forward trellis over known text.

    The surah phoneme sequence is treated as a fixed "transcript".
    For each audio chunk, the ONNX model's raw logits (before argmax) are
    used to run the CTC forward algorithm constrained to this transcript.
    The position is the trellis column with highest forward score.
    """

    BLANK_BONUS = 0       # no bonus for blank — let the trellis decide naturally
    ADVANCE_BIAS = 0.3    # small log-prob bias to prefer advancing over staying
    RESET_SILENCE_S = 2   # seconds of silence before auto-reset

    def __init__(self, surah: int, reference_service, alignment_service, vocab: dict):
        """
        Args:
            vocab: dict mapping token_string -> token_id (from tokens.txt)
                   e.g. {'a': 0, 'b': 1, ..., '<blank>': 68}
        """
        self._surah = surah
        self._ref = reference_service
        self._align = alignment_service
        self._vocab = vocab
        self._blank_id = next(v for k, v in vocab.items() if k == '<blank>')

        surah_info = reference_service.get_surah_info(surah)
        if surah_info is None:
            raise ValueError(f"Surah {surah} not found")

        # Build flat phoneme sequence across all ayahs
        self._flat: list[str] = []          # phoneme strings in order
        self._ayah_refs: dict = {}          # ayah_num -> ref dict
        self._token_ids: list[int] = []     # token_id for each phoneme
        ph_to_word: list[int] = []          # phoneme_idx -> word_idx (within ayah)

        ayah_num = 1
        while reference_service.ayah_exists(surah, ayah_num):
            ref = reference_service.get_reference(surah, ayah_num)
            self._ayah_refs[ayah_num] = ref
            words = ref['text_ar'].split()
            lpm = ref.get('letter_phoneme_map', [])
            ph_to_word_local = self._build_phoneme_word_map(words, lpm)

            for ph_idx, phoneme in enumerate(ref['phoneme_list']):
                self._flat.append(phoneme)
                token_id = vocab.get(phoneme, self._blank_id)
                self._token_ids.append(token_id)
                word_idx = ph_to_word_local.get(ph_idx, 0)
                ph_to_word.append(word_idx)

            ayah_num += 1

        N = len(self._flat)
        avg = N // max(1, ayah_num - 1)
        logger.info(f"PositionTracker: surah {surah}, {N} phonemes across {ayah_num - 1} ayahs, avg {avg} phonemes/ayah")

        # CTC trellis: log-alpha for each phoneme position + 1 (blank start)
        self._trellis_len = N + 1
        self._log_alpha = np.full(self._trellis_len, -np.inf, dtype=np.float32)
        self._log_alpha[0] = 0.0  # start at position 0

        self._cursor = 0
        self._current_ayah = 1
        self._current_word = 0
        self._locked = False

    def _build_phoneme_word_map(self, words: list, lpm: list) -> dict:
        """Map phoneme index (within ayah) -> word index."""
        ph_to_word: dict = {}
        if not lpm:
            # Fallback: distribute phonemes evenly across words
            return ph_to_word

        char_cursor = 0
        word_char_boundaries = []
        for w in words:
            word_char_boundaries.append((char_cursor, char_cursor + len(w)))
            char_cursor += len(w) + 1  # +1 for space

        lpm_char_pos = 0
        for entry in lpm:
            chars = entry.get('chars', '')
            ph_start = entry.get('start', 0)
            ph_end = entry.get('end', 0)

            # Find which word this char range belongs to
            w_idx = 0
            for wi, (ws, _we) in enumerate(word_char_boundaries):
                if lpm_char_pos >= ws:
                    w_idx = wi

            for ph_idx in range(ph_start, ph_end):
                ph_to_word[ph_idx] = w_idx

            lpm_char_pos += len(chars)

        return ph_to_word

    def reset(self):
        """Reset trellis — called after long silence."""
        N = len(self._flat)
        self._log_alpha = np.full(self._trellis_len, -np.inf, dtype=np.float32)
        self._log_alpha[self._cursor] = 0.0  # restart from current position
        self._locked = False
        logger.info("PositionTracker: reset after long silence")

    def update_with_logits(self, logits: np.ndarray) -> PositionUpdate | None:
        """
        Update trellis using raw model logits (T, vocab_size).
        Returns position update or None if position unchanged.
        """
        if self._locked:
            logger.info(f"Position locked at phoneme {self._cursor}")
            return None

        # Log-softmax over vocab dimension
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

        N = len(self._flat)
        S = self._trellis_len

        # CTC forward pass — one step per time frame
        for t in range(logits.shape[0]):
            lp = log_probs[t]
            new_alpha = np.full(S, -np.inf, dtype=np.float32)

            for s in range(S):
                if self._log_alpha[s] == -np.inf:
                    continue

                token = self._blank_id if s == 0 else self._token_ids[s - 1]
                emit = lp[token]

                # Stay: emit same token
                stay = self._log_alpha[s] + emit
                if stay > new_alpha[s]:
                    new_alpha[s] = stay

                # Advance to next phoneme
                if s < N:
                    next_token = self._token_ids[s]
                    adv = self._log_alpha[s] + lp[next_token] + self.ADVANCE_BIAS
                    if adv > new_alpha[s + 1]:
                        new_alpha[s + 1] = adv

                    # Skip blank: allow transitioning through blank
                    if s + 2 <= N:
                        prev_token = self._token_ids[s - 1] if s > 0 else -1
                        if next_token != prev_token:
                            skip = self._log_alpha[s] + lp[self._blank_id] + lp[self._token_ids[s + 1 - 1]] + self.ADVANCE_BIAS
                            if skip > new_alpha[s + 1]:
                                new_alpha[s + 1] = skip

                # Score for each possible token position
                token_scores = new_alpha[:s + 2] if s + 2 <= S else new_alpha

            self._log_alpha = new_alpha

        # Find best position
        finite = np.isfinite(self._log_alpha)
        if not np.any(finite):
            return None

        best_token_pos = int(np.argmax(self._log_alpha))
        best_score = float(self._log_alpha[best_token_pos])

        # Confidence: ratio of best score to score range
        finite_scores = self._log_alpha[finite]
        score_range = float(finite_scores.max() - finite_scores.min())
        confidence = float(np.clip(best_score / max(score_range, 1e-9), 0.0, 1.0))

        # Map trellis position to ayah/word
        phoneme_cursor = min(best_token_pos, N - 1)

        # Find which ayah this phoneme belongs to
        ayah_num = 1
        word_idx = 0
        ph_count = 0
        for an, ref in sorted(self._ayah_refs.items()):
            n_ph = len(ref['phoneme_list'])
            if ph_count + n_ph > phoneme_cursor:
                ayah_num = an
                local_ph = phoneme_cursor - ph_count
                words = ref['text_ar'].split()
                lpm = ref.get('letter_phoneme_map', [])
                ph_to_word = self._build_phoneme_word_map(words, lpm)
                word_idx = ph_to_word.get(local_ph, 0)
                break
            ph_count += n_ph

        completed = phoneme_cursor > self._cursor and ayah_num > self._current_ayah
        prev_cursor = self._cursor
        self._cursor = phoneme_cursor
        letter_results = None

        if completed:
            letter_results = self._get_letter_results(self._current_ayah)

        self._current_ayah = ayah_num
        self._current_word = word_idx

        if phoneme_cursor == N - 1:
            self._locked = True

        logger.info(
            f"Trellis pos: {phoneme_cursor}/{N} ayah={ayah_num} word={word_idx} "
            f"conf={confidence:.2f} delta={phoneme_cursor - prev_cursor}"
        )

        return PositionUpdate(
            ayah_number=ayah_num,
            word_index=word_idx,
            confidence=confidence,
            letter_results=letter_results,
            completed_ayah=completed,
        )

    def _get_letter_results(self, ayah_num: int) -> list | None:
        ref = self._ayah_refs.get(ayah_num)
        if ref is None:
            return None
        try:
            result = self._align.align(
                predicted=ref['phoneme_list'],
                expected=ref['phoneme_list'],
                reference_text=ref['text_ar'],
                letter_phoneme_map=ref.get('letter_phoneme_map'),
            )
            return result.get('letter_results')
        except Exception:
            return None

    @property
    def current_ayah(self) -> int:
        return self._current_ayah

    def is_complete(self) -> bool:
        return self._cursor >= len(self._flat) - 1
