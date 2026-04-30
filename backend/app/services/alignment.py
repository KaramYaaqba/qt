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
    'rˤ': 'r', 'lˤlˤ': 'l', 'sˤsˤ': 's',   # emphatic -> plain
    'aˤ': 'a', 'aˤ:': 'a:',                  # emphatic vowel -> plain
    'bb': 'b', 'dd': 'd', 'ff': 'f',          # geminates -> single
    'kk': 'k', 'll': 'l', 'mm': 'm',
    'nn': 'n', 'rr': 'r', 'ss': 's',
    'tt': 't', 'ww': 'w',
    'a:': 'a', 'i:': 'i', 'u:': 'u',          # long -> short vowels
    'jj': 'j',
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

_SPACE_CHARS = {" ", " ", " "}
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

    def align(
        self,
        predicted: list[str],
        expected: list[str],
        reference_text: str,
        letter_phoneme_map: Optional[list] = None,
    ) -> dict:
        """
        Align predicted phonemes against expected and map to letter positions.

        letter_phoneme_map: list of {chars, phonemes, start, end} from the
        phonemizer's letter_phoneme_mappings(). When provided, errors are mapped
        to the correct letter even when words are skipped.
        """
        if not expected:
            return self._empty_result(reference_text)

        # Normalize before comparison so u: == u:, n == n etc.
        predicted = [_normalize_phoneme(p) for p in predicted]
        expected  = [_normalize_phoneme(e) for e in expected]

        ops = editops(predicted, expected)

        phoneme_errors = []
        for op_type, src_pos, dst_pos in ops:
            error = {
                "type": op_type,
                "position_in_expected": dst_pos,
                "position_in_predicted": src_pos,
            }
            if op_type == "replace":
                error["expected_phoneme"] = expected[dst_pos] if dst_pos < len(expected) else None
                error["got_phoneme"] = predicted[src_pos] if src_pos < len(predicted) else None
            elif op_type == "insert":
                error["expected_phoneme"] = expected[dst_pos] if dst_pos < len(expected) else None
                error["got_phoneme"] = None
            elif op_type == "delete":
                error["expected_phoneme"] = None
                error["got_phoneme"] = predicted[src_pos] if src_pos < len(predicted) else None
            phoneme_errors.append(error)

        error_positions = {e["position_in_expected"] for e in phoneme_errors if e["position_in_expected"] is not None}
        error_map = {e["position_in_expected"]: e for e in phoneme_errors if e["position_in_expected"] is not None}

        if letter_phoneme_map:
            letter_results = self._map_via_letter_phoneme_map(
                reference_text, letter_phoneme_map, error_positions, error_map, expected
            )
        else:
            letter_results = self._map_sequential(
                reference_text, expected, error_positions, error_map
            )

        total_expected = max(len(expected), 1)
        accuracy_phoneme = round((1 - len(phoneme_errors) / total_expected) * 100, 1)

        error_letters = sum(1 for lr in letter_results if lr.get("status") == "error")
        scorable = sum(1 for lr in letter_results if lr.get("status") in ("correct", "error"))
        accuracy_letter = round((1 - error_letters / max(scorable, 1)) * 100, 1)

        return {
            "accuracy_phoneme": accuracy_phoneme,
            "accuracy_letter": accuracy_letter,
            "total_phonemes": len(expected),
            "total_errors": len(phoneme_errors),
            "phoneme_errors": phoneme_errors,
            "letter_results": letter_results,
        }

    def _map_via_letter_phoneme_map(
        self, reference_text, letter_phoneme_map, error_positions, error_map, expected
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
            entry_phonemes = map_entry["phonemes"]

            has_error = any(i in error_positions for i in range(start, end))
            first_err = next(
                (error_map[i] for i in range(start, end) if i in error_positions), None
            )

            for ch in chars:
                # Find actual position in reference_text from char_pos onwards
                actual_pos = reference_text.find(ch, char_pos)
                pos = actual_pos if actual_pos >= 0 else char_pos

                ctype = _char_type(ch)
                if ctype == "diacritic":
                    results.append({"letter": ch, "status": "diacritic", "position": pos})
                elif ctype == "space":
                    results.append({"letter": " ", "status": "space", "position": pos})
                elif ctype == "special":
                    results.append({"letter": ch, "status": "special", "position": pos})
                else:
                    entry = {
                        "letter": ch,
                        "position": pos,
                        "expected_phoneme": entry_phonemes[0] if entry_phonemes else None,
                        "got_phoneme": None,
                        "error_type": None,
                    }
                    if has_error and first_err:
                        entry["status"] = "error"
                        entry["error_type"] = first_err["type"]
                        entry["got_phoneme"] = first_err.get("got_phoneme")
                    else:
                        entry["status"] = "correct"
                    results.append(entry)

                if actual_pos >= 0:
                    char_pos = actual_pos + 1

        return results

    def _map_sequential(
        self, reference_text, expected, error_positions, error_map
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
            entry = {"letter": char, "position": char_idx}
            if phoneme_idx < len(expected):
                entry["expected_phoneme"] = expected[phoneme_idx]
                if phoneme_idx in error_positions:
                    err = error_map[phoneme_idx]
                    entry["status"] = "error"
                    entry["error_type"] = err["type"]
                    entry["got_phoneme"] = err.get("got_phoneme")
                else:
                    entry["status"] = "correct"
            else:
                entry["status"] = "unmapped"
            results.append(entry)
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
