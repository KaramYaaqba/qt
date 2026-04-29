#!/usr/bin/env python3
"""
Prepare Training Data for Quran Speech-to-Phoneme Model

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
JUZ_AMMA_SURAHS = set(range(78, 115))
_TEST_SURAHS = {108, 109, 110, 111, 112, 113, 114}
_DEV_SURAHS  = {104, 105, 106, 107}

# Strip tashkeel only — keeps base Arabic letters intact
_DIACRITICS = re.compile(
    u'[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]'
)
# Normalize alef variants so DB and dataset spellings match
_ALEF = re.compile(r'[آأإٱ]')


def normalize(text: str) -> str:
    text = _ALEF.sub('ا', text)
    text = _DIACRITICS.sub('', text)
    return ' '.join(text.split())


def assign_split(surah: int) -> str:
    if surah in _TEST_SURAHS:
        return "test"
    if surah in _DEV_SURAHS:
        return "dev"
    return "train"


def build_ayah_lookup(db_path: Path) -> dict:
    """Build normalize(ayah_text) -> (surah, ayah) for Juz Amma. O(1) lookup per sample."""
    with open(db_path, encoding="utf-8") as f:
        db = json.load(f)

    ayahs: dict[str, list] = {}
    for entry in db.values():
        surah = int(entry["surah"])
        if surah not in JUZ_AMMA_SURAHS:
            continue
        key = f"{entry['surah']}:{entry['ayah']}"
        ayahs.setdefault(key, []).append((int(entry["word"]), entry["text"]))

    lookup = {}
    for ref, words in ayahs.items():
        words.sort(key=lambda x: x[0])
        full_text = " ".join(w for _, w in words)
        surah, ayah = (int(x) for x in ref.split(":"))
        lookup[normalize(full_text)] = (surah, ayah)

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


def process_sample(sample, idx: int, audio_dir: Path, phonemizer, ayah_lookup: dict):
    match = ayah_lookup.get(normalize(sample.get("text", "")))
    if match is None:
        return None, "no_match"
    surah, ayah = match

    audio     = sample.get("audio", {})
    raw_bytes = audio.get("bytes") if isinstance(audio, dict) else None
    if not raw_bytes:
        return None, "no_audio"
    try:
        audio_raw, sample_rate = sf.read(io.BytesIO(raw_bytes))
    except Exception as e:
        print(f"Warning: audio decode failed sample {idx}: {e}")
        return None, "decode_failed"

    phonemes = get_phonemes(phonemizer, surah, ayah)
    if not phonemes:
        return None, "no_phonemes"

    try:
        audio_array = resample_audio(audio_raw, sample_rate)
    except RuntimeError as e:
        print(f"Error: {e}")
        return None, "resample_failed"

    audio_filename = f"s{surah:03d}_a{ayah:03d}_{idx:06d}.wav"
    audio_path     = audio_dir / audio_filename
    sf.write(audio_path, audio_array, TARGET_SR)

    return {
        "audio_filepath": str(audio_path.absolute()),
        "duration":       round(len(audio_array) / TARGET_SR, 3),
        "text":           phonemes,
        "surah":          surah,
        "ayah":           ayah,
    }, assign_split(surah)


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
    (output_path / "audio").mkdir(exist_ok=True)

    if not HAS_PHONEMIZER:
        print("Error: Phonemizer unavailable.")
        return

    db_path = PHONEMIZER_PATH / "quranic_phonemizer" / "resources" / "Quran.json"
    print("Building Juz Amma ayah lookup table...")
    ayah_lookup = build_ayah_lookup(db_path)
    print(f"Lookup ready: {len(ayah_lookup)} unique ayahs")

    print("Initializing Quranic Phonemizer...")
    phonemizer = Phonemizer()

    print("Loading EveryAyah dataset (streaming)...")
    try:
        dataset = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
        dataset = dataset.cast_column("audio", datasets_Audio(decode=False))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    manifests = {"train": [], "dev": [], "test": []}
    processed = skipped = 0

    for idx, sample in enumerate(dataset):
        entry, result = process_sample(sample, idx, output_path / "audio", phonemizer, ayah_lookup)
        if entry is None:
            skipped += 1
            continue
        manifests[result].append(entry)
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} samples (scanned {idx + 1} total)...")

    print(f"\nDone: {processed} processed, {skipped} skipped")
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
