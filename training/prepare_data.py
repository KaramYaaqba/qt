#!/usr/bin/env python3
"""
Prepare Training Data for Quran Speech-to-Phoneme Model

Streams RetaSy/quranic_audio_dataset (non-Arabic speakers) and generates
phoneme labels for the last 6 surahs of Juz' Amma.

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

# Last 6 surahs only — focused PoC for non-Arabic speaker feedback
TARGET_SURAHS = {109, 110, 111, 112, 113, 114}

# RetaSy uses English surah names — map to surah numbers
SURAH_NAME_TO_NUMBER = {
    "Al-Kafiroon": 109,
    "An-Nasr":     110,
    "Al-Masad":    111,
    "Al-Ikhlas":   112,
    "Al-Falaq":    113,
    "An-Nas":      114,
}

# Only keep recordings labeled as correct or unlabeled (None = not yet reviewed)
KEEP_LABELS = {"correct", None}

# Split: 80% train, 10% dev, 10% test — random by sample index
_DEV_MOD  = 9   # idx % 10 == 9  -> dev
_TEST_MOD = 8   # idx % 10 == 8  -> test

# Strip tashkeel only — keeps base Arabic letters intact
_DIACRITICS = re.compile(
    u'[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]'
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


def build_ayah_lookup(db_path: Path) -> dict:
    """Build normalize(ayah_text) -> (surah, ayah) for target surahs. O(1) per sample."""
    with open(db_path, encoding="utf-8") as f:
        db = json.load(f)

    ayahs: dict[str, list] = {}
    for entry in db.values():
        surah = int(entry["surah"])
        if surah not in TARGET_SURAHS:
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
    """Returns (entry dict, split) on success or (None, reason) on skip."""
    # Filter by surah name
    surah_name = sample.get("Surah", "")
    if surah_name not in SURAH_NAME_TO_NUMBER:
        return None, "out_of_scope"

    # Filter by quality label
    label = sample.get("final_label")
    if label not in KEEP_LABELS:
        return None, f"label_{label}"

    # Match ayah text to get surah:ayah numbers
    ayah_text = sample.get("Aya", "")
    match = ayah_lookup.get(normalize(ayah_text))
    if match is None:
        # Fallback: use surah name mapping + try to infer from text match
        return None, "no_match"
    surah, ayah = match

    # Decode audio
    audio     = sample.get("audio", {})
    raw_bytes = audio.get("bytes") if isinstance(audio, dict) else None
    if not raw_bytes:
        return None, "no_audio"
    try:
        audio_raw, sample_rate = sf.read(io.BytesIO(raw_bytes))
    except Exception as e:
        print(f"Warning: audio decode failed sample {idx}: {e}")
        return None, "decode_failed"

    # Get phoneme labels
    phonemes = get_phonemes(phonemizer, surah, ayah)
    if not phonemes:
        return None, "no_phonemes"

    # Resample to 16kHz
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
    }, assign_split(idx)


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
    print("Building ayah lookup for surahs 109-114...")
    ayah_lookup = build_ayah_lookup(db_path)
    print(f"Lookup ready: {len(ayah_lookup)} unique ayahs")

    print("Initializing Quranic Phonemizer...")
    phonemizer = Phonemizer()

    # Pre-generate all phoneme labels (only 29 ayahs — instant)
    print("Pre-generating phoneme labels for 29 ayahs...")
    phoneme_cache = {}
    for surah in sorted(TARGET_SURAHS):
        for ayah in range(1, 20):  # upper bound; stops when phonemizer raises
            ph = get_phonemes(phonemizer, surah, ayah)
            if not ph:
                break
            phoneme_cache[f"{surah}:{ayah}"] = ph
            print(f"  {surah}:{ayah} -> {ph[:50]}...")
    print(f"Cached {len(phoneme_cache)} ayah phoneme sequences")

    print("\nLoading RetaSy dataset (streaming)...")
    try:
        dataset = load_dataset("RetaSy/quranic_audio_dataset", split="train", streaming=True)
        dataset = dataset.cast_column("audio", datasets_Audio(decode=False))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    manifests = {"train": [], "dev": [], "test": []}
    processed = skipped = 0
    skip_reasons: dict[str, int] = {}

    for idx, sample in enumerate(dataset):
        entry, result = process_sample(sample, idx, output_path / "audio", phonemizer, ayah_lookup)
        if entry is None:
            skipped += 1
            skip_reasons[result] = skip_reasons.get(result, 0) + 1
            continue
        manifests[result].append(entry)
        processed += 1
        print(f"  [{processed}] surah={entry['surah']} ayah={entry['ayah']} "
              f"dur={entry['duration']}s split={result}")

    print(f"\nDone: {processed} processed, {skipped} skipped")
    print("Skip reasons:", skip_reasons)
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
