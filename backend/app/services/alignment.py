"""
Alignment Service

Aligns predicted phonemes against expected phonemes using edit distance,
then maps phoneme-level errors to letter positions in the Arabic text.
"""
from Levenshtein import editops
from typing import Optional

# Phoneme normalization: map equivalent representations to a canonical form
_PHONEME_NORM = str.maketrans({
    'ː': ':',  # Unicode modifier letter colon -> ASCII colon (u:, a:)
    'ɡ': 'g',  # Unicode small letter script g -> ASCII g
    'ñ': 'n',  # nasalized n (Idgham) -> plain n
    'ŋ': 'n',  # velar nasal (Ikhfaa) -> n
})

# Multi-char substitutions handled separately
_PHONEME_MULTI = {
    'm̃': 'm',  # nasalized m (Idgham shafawi) -> m
    'Q': '',   # Qalqala marker -> strip
}

# Beginner mode: collapse distinctions that are very hard for non-Arabic speakers.
# Emphatic consonants -> plain, geminates -> single, long vowels -> short.
_BEGINNER_NORM = {
    # emphatics -> plain (single and geminate forms)
    'rˤ': 'r', 'rˤrˤ': 'r',
    'sˤ': 's', 'sˤsˤ': 's',
    'dˤ': 'd', 'dˤdˤ': 'd',
    'tˤ': 't', 'tˤtˤ': 't',
    'ðˤ': 'ð', 'ðˤðˤ': 'ð',
    'lˤlˤ': 'l',
    'aˤ': 'a', 'aˤ:': 'a:',                  # emphatic vowel -> plain
    # geminates -> single
    'bb': 'b', 'dd': 'd', 'ff': 'f',
    'kk': 'k', 'll': 'l', 'mm': 'm',
    'nn': 'n', 'rr': 'r', 'ss': 's',
    'tt': 't', 'ww': 'w', 'jj': 'j',
    'xx': 'x', 'zz': 'z', 'hh': 'h',
    'qq': 'q', 'ðð': 'ð', 'θθ': 'θ',
    'ʃʃ': 'ʃ', 'ʒʒ': 'ʒ', 'ʕʕ': 'ʕ',
    # long -> short vowels
    'a:': 'a', 'i:': 'i', 'u:': 'u',
    # glottal stop (hamza) — very hard for non-Arabic speakers, skip in beginner
    'ʔ': '',
}

# Set to True to use beginner-friendly scoring
BEGINNER_MODE = True


def _normalize_phoneme(p: str, beginner: bool = BEGINNER_MODE) -> str:
    if p in _PHONEME_MULTI:
        p = _PHONEME_MULTI[p]
        if not p:
            return p
    p = p.translate(_PHONEME_NORM)
    if beginner and p in _BEGINNER_NORM:
        return _BEGINNER_NORM[p]
    return p


# Arabic diacritics (tashkeel) - these don't have their own phonemes
ARABIC_DIACRITICS = {
    "ً",  # Fathatan
    "ٌ",  # Dammatan
    "ٍ",  # Kasratan
    "َ",  # Fatha
    "ُ",  # Damma
    "ِ",  # Kasra
    "ّ",  # Shadda
    "ْ",  # Sukun
    "ٰ",  # Superscript Alef
    "ٖ",  # Subscript Alef
    "ٗ",  # Inverted Damma
    "٘",  # Mark Noon Ghunna
    "ٙ",  # Zwarakay
    "ٚ",  # Vowel Sign Small V Above
    "ٛ",  # Vowel Sign Inverted Small V Above
    "ٜ",  # Vowel Sign Dot Below
    "ٝ",  # Vowel Sign Reversed Damma
    "ٞ",  # Vowel Sign Fatha with Two Dots
    "ٟ",  # Wavy Hamza Below
    "ؗ",  # Zwarakay
    "ؘ",  # Small Fatha
    "ؙ",  # Small Damma
    "ؚ",  # Small Kasra
    "ۡ",  # Small High Dotless Head of Khah
    "ۢ",  # Small High Meem Isolated Form
    "ۣ",  # Small Low Seen
    "ۤ",  # Small High Madda
    "ۥ",  # Small Waw
    "ۦ",  # Small Yeh
    "ۧ",  # Small High Yeh
    "ۨ",  # Small High Noon
    "۪",  # Empty Centre Low Stop
    "۫",  # Empty Centre High Stop
    "۬",  # Rounded High Stop with Filled Centre
    "ۭ",  # Small Low Meem
}

_SPACE_CHARS = {" ", " "}
_SPECIAL_CHARS = set("۝۞۩۫۬")


def _char_type(char: str) -> str:
    if char in ARABIC_DIACRITICS:
        return "diacritic"
    if char in _SPACE_CHARS:
        return "space"
    if char in _SPECIAL_CHARS:
        return "special"
    return "letter"


class AlignmentService:
    """Aligns predicted phonemes against expected and maps errors to letters."""

    def __init__(self, phoneme_letter_map: Optional[dict] = None):
        self.letter_to_phonemes = phoneme_letter_map or {}

    @staticmethod
    def _beginner_pass(predicted: list[str], expected: list[str]):
        """Normalize phonemes for letter-level scoring; return errors + maps."""
        pred_norm = [p for p in (_normalize_phoneme(p, beginner=True) for p in predicted) if p]
        exp_norm  = [_normalize_phoneme(e, beginner=True) for e in expected]
        exp_indexed = [(i, e) for i, e in enumerate(exp_norm) if e]
        exp_clean = [e for _, e in exp_indexed]
        clean_to_orig = {ci: oi for ci, (oi, _) in enumerate(exp_indexed)}

        error_positions: set[int] = set()
        error_map: dict[int, dict] = {}
        phoneme_errors = []
        for op_type, src_pos, dst_pos in editops(pred_norm, exp_clean):
            orig_dst = clean_to_orig.get(dst_pos, dst_pos)
            error: dict = {"type": op_type, "position_in_expected": orig_dst,
                           "position_in_predicted": src_pos,
                           "expected_phoneme": None, "got_phoneme": None}
            if op_type in ("replace", "insert"):
                error["expected_phoneme"] = exp_clean[dst_pos] if dst_pos < len(exp_clean) else None
            if op_type in ("replace", "delete"):
                error["got_phoneme"] = pred_norm[src_pos] if src_pos < len(pred_norm) else None
            error_positions.add(orig_dst)
            error_map[orig_dst] = error
            phoneme_errors.append(error)
        return exp_norm, exp_clean, error_positions, error_map, phoneme_errors

    @staticmethod
    def _tajweed_pass(predicted: list[str], expected: list[str]):
        """Full phoneme alignment for tajweed-level scoring; return error positions + map."""
        pred_full = [_normalize_phoneme(p, beginner=False) for p in predicted]
        exp_full  = [_normalize_phoneme(e, beginner=False) for e in expected]
        error_positions: set[int] = set()
        error_map: dict[int, dict] = {}
        for op_type, src_pos, dst_pos in editops(pred_full, exp_full):
            error_positions.add(dst_pos)
            error_map[dst_pos] = {
                "type": op_type,
                "expected_phoneme": exp_full[dst_pos] if dst_pos < len(exp_full) else None,
                "got_phoneme": pred_full[src_pos] if src_pos < len(pred_full) else None,
            }
        return exp_full, error_positions, error_map

    def align(
        self,
        predicted: list[str],
        expected: list[str],
        reference_text: str,
        letter_phoneme_map: Optional[list] = None,
    ) -> dict:
        """
        Align predicted phonemes against expected and map to letter positions.

        Runs two passes:
        - Beginner pass: normalized phonemes for letter correctness
        - Tajweed pass: full phonemes for tajweed rule checking
        """
        if not expected:
            return self._empty_result(reference_text)

        exp_norm, exp_norm_clean, error_positions_norm, error_map_norm, phoneme_errors = \
            self._beginner_pass(predicted, expected)
        exp_full, error_positions_full, error_map_full = \
            self._tajweed_pass(predicted, expected)

        if letter_phoneme_map:
            letter_results = self._map_via_letter_phoneme_map(
                reference_text, letter_phoneme_map,
                error_positions_norm, error_map_norm,
                error_positions_full, error_map_full,
                exp_norm, exp_full,
            )
        else:
            letter_results = self._map_sequential(
                reference_text, exp_norm, error_positions_norm, error_map_norm,
                error_positions_full, error_map_full, exp_full,
            )

        total_expected = max(len(exp_norm_clean), 1)
        accuracy_phoneme = round((1 - len(phoneme_errors) / total_expected) * 100, 1)
        error_letters = sum(1 for lr in letter_results if lr.get("status") == "error")
        scorable = sum(1 for lr in letter_results if lr.get("status") in ("correct", "error"))
        accuracy_letter = round((1 - error_letters / max(scorable, 1)) * 100, 1)

        return {
            "accuracy_phoneme": accuracy_phoneme,
            "accuracy_letter": accuracy_letter,
            "total_phonemes": len(exp_norm_clean),
            "total_errors": len(phoneme_errors),
            "phoneme_errors": phoneme_errors,
            "letter_results": letter_results,
        }

    def _build_letter_entry(self, ch, pos, entry_phonemes, exp_full_list,
                            has_error, first_err,
                            has_tajweed_error, first_tajweed_err) -> dict:
        ctype = _char_type(ch)
        if ctype == "diacritic":
            return {"letter": ch, "status": "diacritic", "position": pos}
        if ctype == "space":
            return {"letter": " ", "status": "space", "position": pos}
        if ctype == "special":
            return {"letter": ch, "status": "special", "position": pos}

        entry = {
            "letter": ch,
            "position": pos,
            "expected_phoneme": entry_phonemes[0] if entry_phonemes else None,
            "expected_phoneme_full": exp_full_list[0] if exp_full_list else None,
            "got_phoneme": None,
            "got_phoneme_full": None,
            "error_type": None,
            "tajweed_status": "correct",
            "tajweed_error_type": None,
        }
        if has_error and first_err:
            entry["status"] = "error"
            entry["error_type"] = first_err["type"]
            entry["got_phoneme"] = first_err.get("got_phoneme")
        else:
            entry["status"] = "correct"

        if has_tajweed_error and first_tajweed_err:
            entry["tajweed_status"] = "error"
            entry["tajweed_error_type"] = first_tajweed_err["type"]
            entry["got_phoneme_full"] = first_tajweed_err.get("got_phoneme")

        return entry

    def _map_via_letter_phoneme_map(
        self, reference_text, letter_phoneme_map,
        error_positions, error_map,
        error_positions_full, error_map_full,
        expected_norm, expected_full,
    ) -> list:
        """
        Map errors to letters using the phonemizer's letter-phoneme mapping.
        Each entry knows exactly which phoneme indices it owns, so skipped
        words are correctly highlighted even when Levenshtein shifts positions.
        """
        results = []
        char_pos = 0

        for map_entry in letter_phoneme_map:
            chars = map_entry["chars"]
            start = map_entry["start"]
            end = map_entry["end"]
            entry_phonemes_norm = [expected_norm[i] for i in range(start, end) if i < len(expected_norm)]
            entry_phonemes_full = [expected_full[i] for i in range(start, end) if i < len(expected_full)]

            has_error = any(i in error_positions for i in range(start, end))
            first_err = next(
                (error_map[i] for i in range(start, end) if i in error_positions), None
            )
            has_tajweed_error = any(i in error_positions_full for i in range(start, end))
            first_tajweed_err = next(
                (error_map_full[i] for i in range(start, end) if i in error_positions_full), None
            )

            for ch in chars:
                actual_pos = reference_text.find(ch, char_pos)
                pos = actual_pos if actual_pos >= 0 else char_pos
                results.append(self._build_letter_entry(
                    ch, pos, entry_phonemes_norm, entry_phonemes_full,
                    has_error, first_err, has_tajweed_error, first_tajweed_err,
                ))
                if actual_pos >= 0:
                    char_pos = actual_pos + 1

        return results

    def _map_sequential(
        self, reference_text, expected_norm, error_positions, error_map,
        error_positions_full, error_map_full, expected_full,
    ) -> list:
        """Fallback sequential mapping when letter_phoneme_map is not available."""
        results = []
        phoneme_idx = 0
        for char_idx, char in enumerate(reference_text):
            ctype = _char_type(char)
            if ctype == "diacritic":
                results.append({"letter": char, "status": "diacritic", "position": char_idx})
                continue
            if ctype == "space":
                results.append({"letter": " ", "status": "space", "position": char_idx})
                continue
            if ctype == "special":
                results.append({"letter": char, "status": "special", "position": char_idx})
                continue

            norm_ph = expected_norm[phoneme_idx] if phoneme_idx < len(expected_norm) else None
            full_ph = expected_full[phoneme_idx] if phoneme_idx < len(expected_full) else None
            has_error = phoneme_idx in error_positions
            first_err = error_map.get(phoneme_idx)
            has_tajweed_error = phoneme_idx in error_positions_full
            first_tajweed_err = error_map_full.get(phoneme_idx)

            if norm_ph is not None:
                results.append(self._build_letter_entry(
                    char, char_idx, [norm_ph], [full_ph] if full_ph else [],
                    has_error, first_err, has_tajweed_error, first_tajweed_err,
                ))
            else:
                results.append({"letter": char, "status": "unmapped", "position": char_idx})
            phoneme_idx += 1
        return results

    def _empty_result(self, reference_text: str) -> dict:
        results = []
        for char_idx, char in enumerate(reference_text):
            ctype = _char_type(char)
            if ctype == "diacritic":
                status = "diacritic"
            elif ctype == "space":
                status = "space"
            else:
                status = "unmapped"
            results.append({"letter": char, "status": status, "position": char_idx})
        return {
            "accuracy_phoneme": 0.0,
            "accuracy_letter": 0.0,
            "total_phonemes": 0,
            "total_errors": 0,
            "phoneme_errors": [],
            "letter_results": results,
        }


def get_phoneme_description(phoneme: str) -> str:
    descriptions = {
        "b": "ب (ba) - voiced bilabial stop",
        "t": "ت (ta) - voiceless dental stop",
        "dʒ": "ج (jim) - voiced postalveolar affricate",
        "ħ": "ح (ḥa) - voiceless pharyngeal fricative",
        "x": "خ (kha) - voiceless velar fricative",
        "d": "د (dal) - voiced dental stop",
        "ð": "ذ (dhal) - voiced dental fricative",
        "r": "ر (ra) - voiced alveolar trill",
        "z": "ز (zay) - voiced alveolar fricative",
        "s": "س (sin) - voiceless alveolar fricative",
        "ʃ": "ش (shin) - voiceless postalveolar fricative",
        "sˤ": "ص (ṣad) - emphatic s",
        "dˤ": "ض (ḍad) - emphatic d",
        "tˤ": "ط (ṭa) - emphatic t",
        "ðˤ": "ظ (ẓa) - emphatic dh",
        "˕": "ع (ʿayn) - voiced pharyngeal fricative",
        "ʁ": "غ (ghayn) - voiced uvular fricative",
        "f": "ف (fa) - voiceless labiodental fricative",
        "q": "ق (qaf) - voiceless uvular stop",
        "k": "ك (kaf) - voiceless velar stop",
        "l": "ل (lam) - voiced alveolar lateral",
        "m": "م (mim) - voiced bilabial nasal",
        "n": "ن (nun) - voiced alveolar nasal",
        "h": "ه (ha) - voiceless glottal fricative",
        "w": "و (waw) - voiced labial-velar approximant",
        "j": "ي (ya) - voiced palatal approximant",
        "ʔ": "ء (hamza) - glottal stop",
        "a": "فتحة (fatha) - short 'a'",
        "a:": "ألف (alif) - long 'aa'",
        "i": "كسرة (kasra) - short 'i'",
        "i:": "ياء (ya) - long 'ee'",
        "u": "ضمة (damma) - short 'u'",
        "u:": "واو (waw) - long 'oo'",
    }
    return descriptions.get(phoneme, f"Unknown phoneme: {phoneme}")
