#!/usr/bin/env python3
"""
Generate Phoneme Data for Juz' Amma

This script uses the Quranic Phonemizer to generate phoneme transcriptions
for all ayahs in Juz' Amma (Surahs 78-114).

Prerequisites:
    1. Clone the Quranic Phonemizer:
       git clone https://github.com/Hetchy/Quranic-Phonemizer.git phonemizer
    
    2. Install dependencies:
       pip install -r phonemizer/requirements.txt

Usage:
    python generate_phoneme_data.py
    
Output:
    Creates app/data/juz_amma_phonemes.json
"""
import json
import sys
from pathlib import Path

# Add phonemizer to path
PHONEMIZER_PATH = Path(__file__).parent.parent / "phonemizer"
sys.path.insert(0, str(PHONEMIZER_PATH))

try:
    from core.phonemizer import Phonemizer
except ImportError:
    print("ERROR: Quranic-Phonemizer not found!")
    print("Please clone it first:")
    print("  cd backend")
    print("  git clone https://github.com/Hetchy/Quranic-Phonemizer.git phonemizer")
    sys.exit(1)


# Surah names mapping
SURAH_NAMES = {
    78: ("النبأ", "An-Naba"),
    79: ("النازعات", "An-Naziat"),
    80: ("عبس", "Abasa"),
    81: ("التكوير", "At-Takwir"),
    82: ("الانفطار", "Al-Infitar"),
    83: ("المطففين", "Al-Mutaffifin"),
    84: ("الانشقاق", "Al-Inshiqaq"),
    85: ("البروج", "Al-Burooj"),
    86: ("الطارق", "At-Tariq"),
    87: ("الأعلى", "Al-Ala"),
    88: ("الغاشية", "Al-Ghashiyah"),
    89: ("الفجر", "Al-Fajr"),
    90: ("البلد", "Al-Balad"),
    91: ("الشمس", "Ash-Shams"),
    92: ("الليل", "Al-Layl"),
    93: ("الضحى", "Ad-Duha"),
    94: ("الشرح", "Ash-Sharh"),
    95: ("التين", "At-Tin"),
    96: ("العلق", "Al-Alaq"),
    97: ("القدر", "Al-Qadr"),
    98: ("البينة", "Al-Bayyinah"),
    99: ("الزلزلة", "Az-Zalzalah"),
    100: ("العاديات", "Al-Adiyat"),
    101: ("القارعة", "Al-Qariah"),
    102: ("التكاثر", "At-Takathur"),
    103: ("العصر", "Al-Asr"),
    104: ("الهمزة", "Al-Humazah"),
    105: ("الفيل", "Al-Fil"),
    106: ("قريش", "Quraysh"),
    107: ("الماعون", "Al-Maun"),
    108: ("الكوثر", "Al-Kawthar"),
    109: ("الكافرون", "Al-Kafirun"),
    110: ("النصر", "An-Nasr"),
    111: ("المسد", "Al-Masad"),
    112: ("الإخلاص", "Al-Ikhlas"),
    113: ("الفلق", "Al-Falaq"),
    114: ("الناس", "An-Nas"),
}


def main():
    print("Initializing Quranic Phonemizer...")
    pm = Phonemizer()
    
    data = {}
    total_ayahs = 0
    
    for surah_num in range(78, 115):
        name_ar, name_en = SURAH_NAMES[surah_num]
        print(f"Processing Surah {surah_num}: {name_en} ({name_ar})...")
        
        try:
            result = pm.phonemize(str(surah_num), stops=["verse"])
            
            for verse_idx, verse in enumerate(result.verses, 1):
                key = f"{surah_num}:{verse_idx}"
                
                # Get phonemes as a list
                phoneme_str = verse.phonemes_str(phoneme_sep=" ", word_sep=" | ")
                phoneme_list = [p for p in phoneme_str.replace("|", "").split() if p.strip()]
                
                data[key] = {
                    "surah_number": surah_num,
                    "surah_name_ar": name_ar,
                    "surah_name_en": name_en,
                    "ayah_number": verse_idx,
                    "text_ar": verse.text,
                    "phonemes": phoneme_str,
                    "phoneme_list": phoneme_list,
                    "total_phonemes": len(phoneme_list),
                }
                total_ayahs += 1
                
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save to JSON file
    output_path = Path(__file__).parent.parent / "app" / "data" / "juz_amma_phonemes.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! Generated phonemes for {total_ayahs} ayahs")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
