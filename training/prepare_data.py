#!/usr/bin/env python3
"""
Prepare Training Data for Quran Speech-to-Phoneme Model

Downloads the EveryAyah dataset and generates phoneme labels
for training a Conformer-CTC model.

Usage:
    python prepare_data.py [--output_dir ./data]
"""
import json
import argparse
from pathlib import Path
from datasets import load_dataset
import numpy as np
import soundfile as sf
import sys

# Try to import phonemizer (needs to be cloned first)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "phonemizer"))
    from quranic_phonemizer.phonemizer import Phonemizer
    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False
    print("Warning: Quranic Phonemizer not found. Clone it first:")
    print("  cd backend && git clone https://github.com/Hetchy/Quranic-Phonemizer.git phonemizer")


TARGET_SR = 16000

# Juz' Amma surahs
JUZ_AMMA_SURAHS = set(range(78, 115))

# Surah-based split boundaries (within Juz' Amma)
_TEST_SURAHS = {108, 109, 110, 111, 112, 113, 114}
_DEV_SURAHS  = {104, 105, 106, 107}


def assign_split(surah: int) -> str:
    if surah in _TEST_SURAHS:
        return "test"
    if surah in _DEV_SURAHS:
        return "dev"
    return "train"


def resample_audio(audio_array, orig_sr: int):
    """Return audio resampled to TARGET_SR. Raises RuntimeError if librosa missing."""
    if orig_sr == TARGET_SR:
        return np.array(audio_array, dtype=np.float32)
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa is required for resampling. pip install librosa")
    return librosa.resample(
        np.array(audio_array, dtype=np.float32),
        orig_sr=orig_sr,
        target_sr=TARGET_SR,
    )


def get_phonemes(phonemizer, surah: int, ayah: int) -> str:
    """Return space-separated phoneme string, or empty string on failure."""
    if not phonemizer:
        return ""
    try:
        result = phonemizer.phonemize(f"{surah}:{ayah}")
        phonemes = result.phonemes_str(phoneme_sep=" ", word_sep=" ")
        return phonemes.strip() if phonemes else ""
    except Exception as e:
        print(f"Warning: Failed to phonemize {surah}:{ayah}: {e}")
        return ""


def process_sample(sample, idx: int, audio_dir: Path, phonemizer):
    """
    Process one dataset sample.

    Returns (entry dict, split name) on success, or (None, reason) on skip.
    """
    surah = sample.get("surah", 0)
    ayah  = sample.get("ayah", 0)

    if surah not in JUZ_AMMA_SURAHS:
        return None, "out_of_scope"

    audio      = sample.get("audio", {})
    audio_raw  = audio.get("array")
    sample_rate = audio.get("sampling_rate", TARGET_SR)

    if audio_raw is None:
        return None, "no_audio"

    try:
        audio_array = resample_audio(audio_raw, sample_rate)
    except RuntimeError as e:
        print(f"Error: {e}")
        return None, "resample_failed"

    phonemes = get_phonemes(phonemizer, surah, ayah)
    if not phonemes.strip():
        return None, "no_phonemes"

    audio_filename = f"s{surah:03d}_a{ayah:03d}_{idx:06d}.wav"
    audio_path = audio_dir / audio_filename
    sf.write(audio_path, audio_array, TARGET_SR)

    duration = len(audio_array) / TARGET_SR
    entry = {
        "audio_filepath": str(audio_path.absolute()),
        "duration": round(duration, 3),
        "text": phonemes,
        "surah": surah,
        "ayah": ayah,
    }
    return entry, assign_split(surah)


def save_manifests(manifests: dict, output_path: Path):
    for split, entries in manifests.items():
        manifest_path = output_path / f"manifest_{split}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Saved {split} manifest: {len(entries)} samples -> {manifest_path}")


def extract_vocabulary(manifests: dict) -> list:
    """Extract unique phoneme tokens from manifests."""
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

    print("Loading EveryAyah dataset from HuggingFace...")
    try:
        dataset = load_dataset("tarteel-ai/everyayah", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Make sure you have access to the dataset and are logged in:")
        print("  pip install huggingface_hub && huggingface-cli login")
        return

    print(f"Loaded {len(dataset)} samples")

    phonemizer = None
    if HAS_PHONEMIZER:
        print("Initializing Quranic Phonemizer...")
        phonemizer = Phonemizer()
    else:
        print("Error: Phonemizer unavailable — all samples will be skipped.")
        return

    manifests = {"train": [], "dev": [], "test": []}
    processed = skipped = 0

    for idx, sample in enumerate(dataset):
        entry, result = process_sample(sample, idx, audio_dir, phonemizer)
        if entry is None:
            skipped += 1
            continue
        manifests[result].append(entry)
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} samples...")

    print(f"\nProcessed {processed} samples, skipped {skipped}")

    save_manifests(manifests, output_path)

    vocab = extract_vocabulary(manifests)
    vocab_path = output_path / "tokens.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i, token in enumerate(vocab):
            f.write(f"{token} {i}\n")
    print(f"Saved vocabulary: {len(vocab)} tokens -> {vocab_path}")

    print("\nData preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for processed data")
    args = parser.parse_args()
    prepare_data(args.output_dir)
