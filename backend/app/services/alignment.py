"""
Alignment Service

Aligns predicted phonemes against expected phonemes using edit distance,
then maps phoneme-level errors to letter positions in the Arabic text.
"""
from Levenshtein import editops
from typing import Optional


# Arabic diacritics (tashkeel) - these don't have their own phonemes
ARABIC_DIACRITICS = set([
    "\u064B",  # Fathatan
    "\u064C",  # Dammatan
    "\u064D",  # Kasratan
    "\u064E",  # Fatha
    "\u064F",  # Damma
    "\u0650",  # Kasra
    "\u0651",  # Shadda
    "\u0652",  # Sukun
    "\u0670",  # Superscript Alef
    "\u0656",  # Subscript Alef
    "\u0657",  # Inverted Damma
    "\u0658",  # Mark Noon Ghunna
    "\u0659",  # Zwarakay
    "\u065A",  # Vowel Sign Small V Above
    "\u065B",  # Vowel Sign Inverted Small V Above
    "\u065C",  # Vowel Sign Dot Below
    "\u065D",  # Vowel Sign Reversed Damma
    "\u065E",  # Vowel Sign Fatha with Two Dots
    "\u065F",  # Wavy Hamza Below
    "\u0617",  # Zwarakay
    "\u0618",  # Small Fatha
    "\u0619",  # Small Damma
    "\u061A",  # Small Kasra
    "\u06E1",  # Small High Dotless Head of Khah
    "\u06E2",  # Small High Meem Isolated Form
    "\u06E3",  # Small Low Seen
    "\u06E4",  # Small High Madda
    "\u06E5",  # Small Waw
    "\u06E6",  # Small Yeh
    "\u06E7",  # Small High Yeh
    "\u06E8",  # Small High Noon
    "\u06EA",  # Empty Centre Low Stop
    "\u06EB",  # Empty Centre High Stop
    "\u06EC",  # Rounded High Stop with Filled Centre
    "\u06ED",  # Small Low Meem
])


class AlignmentService:
    """
    Aligns predicted phonemes against expected phonemes and maps errors to letters.
    """
    
    def __init__(self, phoneme_letter_map: Optional[dict] = None):
        """
        Initialize alignment service.
        
        Args:
            phoneme_letter_map: Optional mapping from phonemes to Arabic letters
        """
        self.letter_to_phonemes = phoneme_letter_map or {}

    def align(
        self, 
        predicted: list[str], 
        expected: list[str], 
        reference_text: str
    ) -> dict:
        """
        Align predicted phonemes against expected and map to letter positions.
        
        Args:
            predicted: List of phonemes from speech recognition
            expected: List of expected phonemes from reference
            reference_text: Arabic text of the ayah
            
        Returns:
            Dictionary with alignment results including:
            - accuracy_phoneme: Overall phoneme-level accuracy %
            - accuracy_letter: Letter-level accuracy %
            - total_phonemes: Total expected phonemes
            - total_errors: Number of phoneme errors
            - phoneme_errors: List of detailed phoneme errors
            - letter_results: List of per-letter results with status
        """
        # Handle empty inputs
        if not expected:
            return self._empty_result(reference_text)
            
        # Compute edit operations between predicted and expected sequences
        ops = editops(predicted, expected)

        # Build list of phoneme-level errors
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
                # Missing phoneme - expected but not produced
                error["expected_phoneme"] = expected[dst_pos] if dst_pos < len(expected) else None
                error["got_phoneme"] = None
            elif op_type == "delete":
                # Extra phoneme - produced but not expected
                error["expected_phoneme"] = None
                error["got_phoneme"] = predicted[src_pos] if src_pos < len(predicted) else None
                
            phoneme_errors.append(error)

        # Build set of error positions for quick lookup
        error_positions = {e["position_in_expected"] for e in phoneme_errors if e["position_in_expected"] is not None}
        error_map = {e["position_in_expected"]: e for e in phoneme_errors if e["position_in_expected"] is not None}

        # Map phoneme errors to letter positions
        letter_results = []
        phoneme_idx = 0
        
        for char_idx, char in enumerate(reference_text):
            # Handle diacritics - they inherit status from their base letter
            if char in ARABIC_DIACRITICS:
                letter_results.append({
                    "letter": char,
                    "status": "diacritic",
                    "position": char_idx,
                })
                continue
                
            # Handle spaces
            if char == " " or char == "\u00A0":  # regular space or non-breaking space
                letter_results.append({
                    "letter": " ",
                    "status": "space",
                    "position": char_idx,
                })
                continue
                
            # Handle special Quranic characters (Waqf marks, etc.)
            if char in "۝۞۩۫۬":
                letter_results.append({
                    "letter": char,
                    "status": "special",
                    "position": char_idx,
                })
                continue

            # Regular letter
            entry = {
                "letter": char,
                "position": char_idx,
            }
            
            # Map to phoneme if we have more phonemes
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
                # More letters than phonemes (shouldn't normally happen)
                entry["status"] = "unmapped"
                
            letter_results.append(entry)
            phoneme_idx += 1

        # Calculate accuracies
        total_expected = max(len(expected), 1)
        accuracy_phoneme = round((1 - len(phoneme_errors) / total_expected) * 100, 1)
        
        error_letters = sum(1 for lr in letter_results if lr.get("status") == "error")
        scorable_letters = sum(1 for lr in letter_results if lr.get("status") in ("correct", "error"))
        accuracy_letter = round((1 - error_letters / max(scorable_letters, 1)) * 100, 1)

        return {
            "accuracy_phoneme": accuracy_phoneme,
            "accuracy_letter": accuracy_letter,
            "total_phonemes": len(expected),
            "total_errors": len(phoneme_errors),
            "phoneme_errors": phoneme_errors,
            "letter_results": letter_results,
        }
    
    def _empty_result(self, reference_text: str) -> dict:
        """Generate result for empty expected phonemes."""
        letter_results = []
        for char_idx, char in enumerate(reference_text):
            if char in ARABIC_DIACRITICS:
                status = "diacritic"
            elif char == " ":
                status = "space"
            else:
                status = "unmapped"
            letter_results.append({
                "letter": char,
                "status": status,
                "position": char_idx,
            })
            
        return {
            "accuracy_phoneme": 0.0,
            "accuracy_letter": 0.0,
            "total_phonemes": 0,
            "total_errors": 0,
            "phoneme_errors": [],
            "letter_results": letter_results,
        }


def get_phoneme_description(phoneme: str) -> str:
    """
    Get human-readable description of a phoneme.
    
    Args:
        phoneme: IPA phoneme symbol
        
    Returns:
        Description string
    """
    descriptions = {
        # Consonants
        "b": "ب (ba) - voiced bilabial stop",
        "t": "ت (ta) - voiceless dental stop",
        "θ": "ث (tha) - voiceless dental fricative",
        "dʒ": "ج (jim) - voiced postalveolar affricate",
        "ħ": "ح (ḥa) - voiceless pharyngeal fricative",
        "x": "خ (kha) - voiceless velar fricative",
        "d": "د (dal) - voiced dental stop",
        "ð": "ذ (dhal) - voiced dental fricative",
        "r": "ر (ra) - voiced alveolar trill",
        "z": "ز (zay) - voiced alveolar fricative",
        "s": "س (sin) - voiceless alveolar fricative",
        "ʃ": "ش (shin) - voiceless postalveolar fricative",
        "sˤ": "ص (ṣad) - emphatic voiceless alveolar fricative",
        "dˤ": "ض (ḍad) - emphatic voiced dental stop",
        "tˤ": "ط (ṭa) - emphatic voiceless dental stop",
        "ðˤ": "ظ (ẓa) - emphatic voiced dental fricative",
        "ʕ": "ع (ʿayn) - voiced pharyngeal fricative",
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
        # Vowels
        "a": "فتحة (fatha) - short 'a'",
        "aː": "ألف (alif) - long 'aa'",
        "i": "كسرة (kasra) - short 'i'",
        "iː": "ياء (ya) - long 'ee'",
        "u": "ضمة (damma) - short 'u'",
        "uː": "واو (waw) - long 'oo'",
    }
    return descriptions.get(phoneme, f"Unknown phoneme: {phoneme}")
