#!/usr/bin/env python3
"""
Build Letter-to-Phoneme Mapping

Creates a JSON mapping from Arabic letters to their possible phonemes.
This is used by the alignment service to provide helpful error messages.

Usage:
    python build_letter_phoneme_map.py
"""
import json
from pathlib import Path


# Mapping of Arabic letters to their IPA phonemes
# Note: Some letters can produce multiple phonemes depending on context
LETTER_PHONEME_MAP = {
    # Consonants
    "ء": {
        "phonemes": ["ʔ"],
        "name_ar": "همزة",
        "name_en": "hamza",
        "description": "glottal stop"
    },
    "ب": {
        "phonemes": ["b"],
        "name_ar": "باء",
        "name_en": "ba",
        "description": "voiced bilabial stop"
    },
    "ت": {
        "phonemes": ["t"],
        "name_ar": "تاء",
        "name_en": "ta",
        "description": "voiceless dental stop"
    },
    "ث": {
        "phonemes": ["θ"],
        "name_ar": "ثاء",
        "name_en": "tha",
        "description": "voiceless dental fricative"
    },
    "ج": {
        "phonemes": ["dʒ"],
        "name_ar": "جيم",
        "name_en": "jim",
        "description": "voiced postalveolar affricate"
    },
    "ح": {
        "phonemes": ["ħ"],
        "name_ar": "حاء",
        "name_en": "ha (emphatic)",
        "description": "voiceless pharyngeal fricative"
    },
    "خ": {
        "phonemes": ["x"],
        "name_ar": "خاء",
        "name_en": "kha",
        "description": "voiceless velar fricative"
    },
    "د": {
        "phonemes": ["d"],
        "name_ar": "دال",
        "name_en": "dal",
        "description": "voiced dental stop"
    },
    "ذ": {
        "phonemes": ["ð"],
        "name_ar": "ذال",
        "name_en": "dhal",
        "description": "voiced dental fricative"
    },
    "ر": {
        "phonemes": ["r"],
        "name_ar": "راء",
        "name_en": "ra",
        "description": "voiced alveolar trill"
    },
    "ز": {
        "phonemes": ["z"],
        "name_ar": "زاي",
        "name_en": "zay",
        "description": "voiced alveolar fricative"
    },
    "س": {
        "phonemes": ["s"],
        "name_ar": "سين",
        "name_en": "sin",
        "description": "voiceless alveolar fricative"
    },
    "ش": {
        "phonemes": ["ʃ"],
        "name_ar": "شين",
        "name_en": "shin",
        "description": "voiceless postalveolar fricative"
    },
    "ص": {
        "phonemes": ["sˤ"],
        "name_ar": "صاد",
        "name_en": "sad",
        "description": "emphatic s"
    },
    "ض": {
        "phonemes": ["dˤ"],
        "name_ar": "ضاد",
        "name_en": "dad",
        "description": "emphatic d"
    },
    "ط": {
        "phonemes": ["tˤ"],
        "name_ar": "طاء",
        "name_en": "ta (emphatic)",
        "description": "emphatic t"
    },
    "ظ": {
        "phonemes": ["ðˤ"],
        "name_ar": "ظاء",
        "name_en": "za (emphatic)",
        "description": "emphatic dh"
    },
    "ع": {
        "phonemes": ["ʕ"],
        "name_ar": "عين",
        "name_en": "ayn",
        "description": "voiced pharyngeal fricative"
    },
    "غ": {
        "phonemes": ["ʁ"],
        "name_ar": "غين",
        "name_en": "ghayn",
        "description": "voiced uvular fricative"
    },
    "ف": {
        "phonemes": ["f"],
        "name_ar": "فاء",
        "name_en": "fa",
        "description": "voiceless labiodental fricative"
    },
    "ق": {
        "phonemes": ["q"],
        "name_ar": "قاف",
        "name_en": "qaf",
        "description": "voiceless uvular stop"
    },
    "ك": {
        "phonemes": ["k"],
        "name_ar": "كاف",
        "name_en": "kaf",
        "description": "voiceless velar stop"
    },
    "ل": {
        "phonemes": ["l"],
        "name_ar": "لام",
        "name_en": "lam",
        "description": "voiced alveolar lateral"
    },
    "م": {
        "phonemes": ["m"],
        "name_ar": "ميم",
        "name_en": "mim",
        "description": "voiced bilabial nasal"
    },
    "ن": {
        "phonemes": ["n"],
        "name_ar": "نون",
        "name_en": "nun",
        "description": "voiced alveolar nasal"
    },
    "ه": {
        "phonemes": ["h"],
        "name_ar": "هاء",
        "name_en": "ha",
        "description": "voiceless glottal fricative"
    },
    "و": {
        "phonemes": ["w", "uː"],
        "name_ar": "واو",
        "name_en": "waw",
        "description": "labial-velar approximant or long u"
    },
    "ي": {
        "phonemes": ["j", "iː"],
        "name_ar": "ياء",
        "name_en": "ya",
        "description": "palatal approximant or long i"
    },
    # Special letters
    "ا": {
        "phonemes": ["aː"],
        "name_ar": "ألف",
        "name_en": "alif",
        "description": "long a vowel"
    },
    "آ": {
        "phonemes": ["ʔ", "aː"],
        "name_ar": "ألف مدة",
        "name_en": "alif madda",
        "description": "hamza + long a"
    },
    "أ": {
        "phonemes": ["ʔ"],
        "name_ar": "ألف همزة",
        "name_en": "alif hamza",
        "description": "hamza on alif"
    },
    "إ": {
        "phonemes": ["ʔ"],
        "name_ar": "ألف همزة تحتية",
        "name_en": "alif hamza below",
        "description": "hamza below alif"
    },
    "ؤ": {
        "phonemes": ["ʔ"],
        "name_ar": "واو همزة",
        "name_en": "waw hamza",
        "description": "hamza on waw"
    },
    "ئ": {
        "phonemes": ["ʔ"],
        "name_ar": "ياء همزة",
        "name_en": "ya hamza",
        "description": "hamza on ya"
    },
    "ة": {
        "phonemes": ["h", "t"],
        "name_ar": "تاء مربوطة",
        "name_en": "ta marbuta",
        "description": "h when pausing, t when continuing"
    },
    "ى": {
        "phonemes": ["aː"],
        "name_ar": "ألف مقصورة",
        "name_en": "alif maqsura",
        "description": "long a vowel"
    },
}


def main():
    output_path = Path(__file__).parent.parent / "app" / "data" / "phoneme_to_letter_map.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(LETTER_PHONEME_MAP, f, ensure_ascii=False, indent=2)
    
    print(f"Created letter-phoneme mapping for {len(LETTER_PHONEME_MAP)} letters")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
