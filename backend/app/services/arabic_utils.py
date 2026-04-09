"""
Arabic Text Utilities

Functions for Arabic text normalization and letter-phoneme mapping.
"""
import re
from typing import Optional

# Arabic letter categories
ARABIC_LETTERS = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")

# Arabic diacritics (tashkeel/harakat)
ARABIC_DIACRITICS = set([
    "\u064B",  # Fathatan (ً)
    "\u064C",  # Dammatan (ٌ)
    "\u064D",  # Kasratan (ٍ)
    "\u064E",  # Fatha (َ)
    "\u064F",  # Damma (ُ)
    "\u0650",  # Kasra (ِ)
    "\u0651",  # Shadda (ّ)
    "\u0652",  # Sukun (ْ)
    "\u0670",  # Dagger Alif (ٰ)
])

# Extended Quranic marks
QURANIC_MARKS = set([
    "\u0656",  # Subscript Alef
    "\u0657",  # Inverted Damma
    "\u0658",  # Mark Noon Ghunna
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
    "\u06EC",  # Rounded High Stop
    "\u06ED",  # Small Low Meem
])

# Waqf (pause) signs
WAQF_SIGNS = set("۝۞۩ۖۗۘۙۚۛ")

# Letter name mapping
LETTER_NAMES = {
    "ء": ("همزة", "hamza"),
    "آ": ("ألف مدّة", "alif madda"),
    "أ": ("ألف همزة", "alif hamza"),
    "ؤ": ("واو همزة", "waw hamza"),
    "إ": ("ألف همزة تحتية", "alif hamza below"),
    "ئ": ("ياء همزة", "ya hamza"),
    "ا": ("ألف", "alif"),
    "ب": ("باء", "ba"),
    "ة": ("تاء مربوطة", "ta marbuta"),
    "ت": ("تاء", "ta"),
    "ث": ("ثاء", "tha"),
    "ج": ("جيم", "jim"),
    "ح": ("حاء", "ha"),
    "خ": ("خاء", "kha"),
    "د": ("دال", "dal"),
    "ذ": ("ذال", "dhal"),
    "ر": ("راء", "ra"),
    "ز": ("زاي", "zay"),
    "س": ("سين", "sin"),
    "ش": ("شين", "shin"),
    "ص": ("صاد", "sad"),
    "ض": ("ضاد", "dad"),
    "ط": ("طاء", "ta emphatic"),
    "ظ": ("ظاء", "za emphatic"),
    "ع": ("عين", "ayn"),
    "غ": ("غين", "ghayn"),
    "ف": ("فاء", "fa"),
    "ق": ("قاف", "qaf"),
    "ك": ("كاف", "kaf"),
    "ل": ("لام", "lam"),
    "م": ("ميم", "mim"),
    "ن": ("نون", "nun"),
    "ه": ("هاء", "ha"),
    "و": ("واو", "waw"),
    "ى": ("ألف مقصورة", "alif maqsura"),
    "ي": ("ياء", "ya"),
}

# Basic letter to phoneme mapping (simplified)
# Note: This is context-independent. Real pronunciation depends on diacritics and position.
LETTER_TO_PHONEME = {
    "ء": ["ʔ"],
    "آ": ["ʔ", "aː"],
    "أ": ["ʔ"],
    "ؤ": ["ʔ"],
    "إ": ["ʔ"],
    "ئ": ["ʔ"],
    "ا": ["aː"],  # or silent
    "ب": ["b"],
    "ة": ["h", "t"],  # depends on whether pausing
    "ت": ["t"],
    "ث": ["θ"],
    "ج": ["dʒ"],
    "ح": ["ħ"],
    "خ": ["x"],
    "د": ["d"],
    "ذ": ["ð"],
    "ر": ["r"],
    "ز": ["z"],
    "س": ["s"],
    "ش": ["ʃ"],
    "ص": ["sˤ"],
    "ض": ["dˤ"],
    "ط": ["tˤ"],
    "ظ": ["ðˤ"],
    "ع": ["ʕ"],
    "غ": ["ʁ"],
    "ف": ["f"],
    "ق": ["q"],
    "ك": ["k"],
    "ل": ["l"],
    "م": ["m"],
    "ن": ["n"],
    "ه": ["h"],
    "و": ["w", "uː"],  # depends on diacritics
    "ى": ["aː"],
    "ي": ["j", "iː"],  # depends on diacritics
}


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for display.
    
    - Normalizes different forms of alif
    - Normalizes different forms of ya
    - Normalizes different forms of hamza
    
    Args:
        text: Arabic text
        
    Returns:
        Normalized text
    """
    # Normalize alif variants
    text = re.sub("[إأآا]", "ا", text)
    
    # Normalize ya variants (alif maqsura to ya)
    text = text.replace("ى", "ي")
    
    # Normalize ta marbuta to ha
    text = text.replace("ة", "ه")
    
    return text


def strip_diacritics(text: str) -> str:
    """
    Remove all diacritics from Arabic text.
    
    Args:
        text: Arabic text with diacritics
        
    Returns:
        Text without diacritics
    """
    return "".join(c for c in text if c not in ARABIC_DIACRITICS and c not in QURANIC_MARKS)


def is_arabic_letter(char: str) -> bool:
    """Check if a character is an Arabic letter."""
    return char in ARABIC_LETTERS


def is_diacritic(char: str) -> bool:
    """Check if a character is an Arabic diacritic."""
    return char in ARABIC_DIACRITICS or char in QURANIC_MARKS


def is_waqf_sign(char: str) -> bool:
    """Check if a character is a Quranic waqf (pause) sign."""
    return char in WAQF_SIGNS


def get_letter_name(letter: str, language: str = "ar") -> Optional[str]:
    """
    Get the name of an Arabic letter.
    
    Args:
        letter: Arabic letter
        language: "ar" for Arabic name, "en" for English
        
    Returns:
        Letter name or None if not found
    """
    if letter not in LETTER_NAMES:
        return None
    ar_name, en_name = LETTER_NAMES[letter]
    return ar_name if language == "ar" else en_name


def get_letter_phonemes(letter: str) -> list[str]:
    """
    Get possible phonemes for an Arabic letter.
    
    Note: This is context-independent. Actual pronunciation
    depends on surrounding letters and diacritics.
    
    Args:
        letter: Arabic letter
        
    Returns:
        List of possible phonemes
    """
    return LETTER_TO_PHONEME.get(letter, [])


def split_into_words(text: str) -> list[str]:
    """
    Split Arabic text into words, preserving diacritics with their letters.
    
    Args:
        text: Arabic text
        
    Returns:
        List of words
    """
    return text.split()


def get_letter_with_diacritics(text: str, position: int) -> str:
    """
    Get a letter along with its following diacritics.
    
    Args:
        text: Arabic text
        position: Position of the base letter
        
    Returns:
        Letter with its diacritics
    """
    if position >= len(text):
        return ""
        
    result = text[position]
    i = position + 1
    
    while i < len(text) and is_diacritic(text[i]):
        result += text[i]
        i += 1
        
    return result


def count_letters(text: str) -> int:
    """
    Count actual Arabic letters (excluding diacritics and spaces).
    
    Args:
        text: Arabic text
        
    Returns:
        Number of letters
    """
    return sum(1 for c in text if is_arabic_letter(c))
