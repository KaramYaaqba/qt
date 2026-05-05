#!/usr/bin/env python3
"""
Prepare Training Data for Quran Speech-to-Phoneme Model

Streams two datasets:
1. RetaSy/quranic_audio_dataset — non-Arabic speakers (primary)
2. tarteel-ai/everyayah — professional reciters (supplementary)

Targets the last 20 surahs of Juz' Amma (95–114).

Usage:
    python prepare_data.py [--output_dir ./data]
"""
import io
import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset, Audio as datasets_Audio
import numpy as np
import soundfile as sf
import sys

PHONEMIZER_PATH = Path(__file__).parent.parent / "backend" / "phonemizer"
sys.path.insert(0, str(PHONEMIZER_PATH))

try:
    from quranic_phonemizer.phonemizer import Phonemizer
    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False
    print("Warning: Quranic Phonemizer not found.")
    print("  cd backend && git clone https://github.com/Hetchy/Quranic-Phonemizer.git phonemizer")

TARGET_SR = 16000

# Last 2 Juz's: Juz' 29 (67-77) + Juz' 30 / Juz' Amma (78-114)
TARGET_SURAHS = set(range(67, 115))

# RetaSy English surah name -> surah number
SURAH_NAME_TO_NUMBER = {
    # Juz' 29 (67-77)
    "Al-Mulk":      67,
    "Al-Qalam":     68,
    "Al-Haqqah":    69,
    "Al-Maarij":    70,
    "Nuh":          71,
    "Al-Jinn":      72,
    "Al-Muzzammil": 73,
    "Al-Muddaththir": 74,
    "Al-Qiyamah":   75,
    "Al-Insan":     76,
    "Al-Mursalat":  77,
    # Juz' 30 / Juz' Amma (78-114)
    "An-Naba":      78,
    "An-Naziat":    79,
    "Abasa":        80,
    "At-Takwir":    81,
    "Al-Infitar":   82,
    "Al-Mutaffifin": 83,
    "Al-Inshiqaq":  84,
    "Al-Burooj":    85,
    "At-Tariq":     86,
    "Al-Ala":       87,
    "Al-Ghashiyah": 88,
    "Al-Fajr":      89,
    "Al-Balad":     90,
    "Ash-Shams":    91,
    "Al-Layl":      92,
    "Ad-Duha":      93,
    "Ash-Sharh":    94,
    "Al-Tin":       95,
    "Al-Alaq":      96,
    "Al-Qadr":      97,
    "Al-Bayyinah":  98,
    "Az-Zalzalah":  99,
    "Al-Adiyat":    100,
    "Al-Qariah":    101,
    "At-Takathur":  102,
    "Al-Asr":       103,
    "Al-Humazah":   104,
    "Al-Fil":       105,
    "Quraish":      106,
    "Al-Maaoon":    107,
    "Al-Kauthar":   108,
    "Al-Kafiroon":  109,
    "An-Nasr":      110,
    "Al-Masad":     111,
    "Al-Ikhlas":    112,
    "Al-Falaq":     113,
    "An-Nas":       114,
}

# Only keep recordings labeled as correct or unlabeled
KEEP_LABELS = {"correct", None}

_DEV_MOD  = 9
_TEST_MOD = 8

_DIACRITICS = re.compile(
    u'[ً-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]'
)
_ALEF = re.compile(r'[آأإٱ]')


def normalize(text: str) -> str:
    text = _ALEF.sub('ا', text)
    text = _DIACRITICS.sub('', text)
    return ' '.join(text.split())


def assign_split(idx: int) -> str:
    r = idx % 10
    if r == _TEST_MOD:
        return "test"
    if r == _DEV_MOD:
        return "dev"
    return "train"


def build_ayah_lookup(phonemizer) -> dict:
    """Build normalize(ayah_text) -> (surah, ayah) for target surahs."""
    lookup = {}
    for surah in sorted(TARGET_SURAHS):
        for ayah in range(1, 30):
            try:
                result = phonemizer.phonemize(f"{surah}:{ayah}")
                text = result._text
                if not text:
                    break
                lookup[normalize(text)] = (surah, ayah)
            except Exception:
                break
    return lookup


def resample_audio(audio_array, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return np.array(audio_array, dtype=np.float32)
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa required: pip install librosa")
    return librosa.resample(np.array(audio_array, dtype=np.float32),
                            orig_sr=orig_sr, target_sr=TARGET_SR)


def get_phonemes(phonemizer, surah: int, ayah: int) -> str:
    try:
        result = phonemizer.phonemize(f"{surah}:{ayah}")
        phonemes = result.phonemes_str(phoneme_sep=" ", word_sep=" ")
        return phonemes.strip() if phonemes else ""
    except Exception as e:
        print(f"Warning: phonemize({surah}:{ayah}) failed: {e}")
        return ""


def decode_audio(audio: dict, idx: int):
    """Decode raw audio bytes to numpy array + sample rate."""
    raw_bytes = audio.get("bytes") if isinstance(audio, dict) else None
    if not raw_bytes:
        return None, None
    try:
        arr, sr = sf.read(io.BytesIO(raw_bytes))
        return arr, sr
    except Exception as e:
        print(f"Warning: audio decode failed sample {idx}: {e}")
        return None, None


def make_entry(audio_raw, sample_rate, surah, ayah, phonemes, idx, audio_dir):
    try:
        audio_array = resample_audio(audio_raw, sample_rate)
    except RuntimeError as e:
        print(f"Error: {e}")
        return None
    audio_path = audio_dir / f"s{surah:03d}_a{ayah:03d}_{idx:06d}.wav"
    sf.write(audio_path, audio_array, TARGET_SR)
    return {
        "audio_filepath": str(audio_path.absolute()),
        "duration":       round(len(audio_array) / TARGET_SR, 3),
        "text":           phonemes,
        "surah":          surah,
        "ayah":           ayah,
    }


def process_retasy(sample, idx, audio_dir, phonemizer, ayah_lookup):
    surah_name = sample.get("Surah", "")
    if surah_name not in SURAH_NAME_TO_NUMBER:
        return None, "out_of_scope"
    if sample.get("final_label") not in KEEP_LABELS:
        return None, f"label_{sample.get('final_label')}"
    match = ayah_lookup.get(normalize(sample.get("Aya", "")))
    if match is None:
        return None, "no_match"
    surah, ayah = match
    audio_raw, sr = decode_audio(sample.get("audio", {}), idx)
    if audio_raw is None:
        return None, "no_audio"
    phonemes = get_phonemes(phonemizer, surah, ayah)
    if not phonemes:
        return None, "no_phonemes"
    entry = make_entry(audio_raw, sr, surah, ayah, phonemes, idx, audio_dir)
    return (entry, assign_split(idx)) if entry else (None, "resample_failed")


def process_everyayah(sample, idx, audio_dir, phonemizer, ayah_lookup):
    match = ayah_lookup.get(normalize(sample.get("text", "")))
    if match is None:
        return None, "no_match"
    surah, ayah = match
    audio_raw, sr = decode_audio(sample.get("audio", {}), idx)
    if audio_raw is None:
        return None, "no_audio"
    phonemes = get_phonemes(phonemizer, surah, ayah)
    if not phonemes:
        return None, "no_phonemes"
    entry = make_entry(audio_raw, sr, surah, ayah, phonemes, idx, audio_dir)
    return (entry, assign_split(idx)) if entry else (None, "resample_failed")


def save_manifests(manifests: dict, output_path: Path):
    for split, entries in manifests.items():
        path = output_path / f"manifest_{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Saved {split}: {len(entries)} samples -> {path}")


def extract_vocabulary(manifests: dict) -> list:
    tokens = set()
    for entries in manifests.values():
        for entry in entries:
            tokens.update(entry.get("text", "").split())
    vocab = sorted(tokens)
    vocab.append("<blank>")
    return vocab


def prepare_data(output_dir: str = "./data"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    if not HAS_PHONEMIZER:
        print("Error: Phonemizer unavailable.")
        return

    print("Initializing Quranic Phonemizer...")
    phonemizer = Phonemizer()

    print(f"Building ayah lookup for surahs {min(TARGET_SURAHS)}-{max(TARGET_SURAHS)}...")
    ayah_lookup = build_ayah_lookup(phonemizer)
    print(f"Lookup ready: {len(ayah_lookup)} unique ayahs")

    manifests = {"train": [], "dev": [], "test": []}
    processed = skipped = 0

    # --- Dataset 1: RetaSy (non-native speakers) ---
    print("\nLoading RetaSy dataset (non-native speakers)...")
    try:
        ds_retasy = load_dataset("RetaSy/quranic_audio_dataset", split="train", streaming=True)
        ds_retasy = ds_retasy.cast_column("audio", datasets_Audio(decode=False))
        for idx, sample in enumerate(ds_retasy):
            entry, result = process_retasy(sample, idx, audio_dir, phonemizer, ayah_lookup)
            if entry is None:
                skipped += 1
                continue
            manifests[result].append(entry)
            processed += 1
            if processed % 100 == 0:
                print(f"  RetaSy: {processed} processed (scanned {idx+1} total)...")
    except Exception as e:
        print(f"RetaSy load failed: {e}")

    retasy_count = processed
    print(f"RetaSy done: {retasy_count} samples")

    # --- Dataset 2: everyayah (professional reciters, supplementary) ---
    print("\nLoading everyayah dataset (professional reciters)...")
    try:
        ds_every = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
        ds_every = ds_every.cast_column("audio", datasets_Audio(decode=False))
        every_processed = 0
        for idx, sample in enumerate(ds_every):
            # Use offset idx to avoid filename collisions with RetaSy
            entry, result = process_everyayah(sample, idx + 500000, audio_dir, phonemizer, ayah_lookup)
            if entry is None:
                skipped += 1
                continue
            manifests[result].append(entry)
            processed += 1
            every_processed += 1
            if every_processed % 500 == 0:
                print(f"  everyayah: {every_processed} processed (scanned {idx+1} total)...")
    except Exception as e:
        print(f"everyayah load failed: {e}")

    print(f"everyayah done: {processed - retasy_count} samples")
    print(f"\nTotal: {processed} processed, {skipped} skipped")

    save_manifests(manifests, output_path)

    vocab = extract_vocabulary(manifests)
    vocab_path = output_path / "tokens.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i, token in enumerate(vocab):
            f.write(f"{token} {i}\n")
    print(f"Vocabulary: {len(vocab)} tokens -> {vocab_path}")
    print("\nData preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data")
    args = parser.parse_args()
    prepare_data(args.output_dir)
