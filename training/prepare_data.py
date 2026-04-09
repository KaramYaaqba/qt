#!/usr/bin/env python3
"""
Prepare Training Data for Quran Speech-to-Phoneme Model

Downloads the EveryAyah dataset and generates phoneme labels
for training a Conformer-CTC model.

Usage:
    python prepare_data.py [--output_dir ./data]
"""
import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
import sys

# Try to import phonemizer (needs to be cloned first)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "phonemizer"))
    from core.phonemizer import Phonemizer
    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False
    print("Warning: Quranic Phonemizer not found. Clone it first:")
    print("  cd backend && git clone https://github.com/Hetchy/Quranic-Phonemizer.git phonemizer")


# Juz' Amma surahs
JUZ_AMMA_SURAHS = list(range(78, 115))


def prepare_data(output_dir: str = "./data"):
    """
    Download and prepare training data.
    
    Args:
        output_dir: Directory to save processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    print("Loading EveryAyah dataset from HuggingFace...")
    # Note: You may need to authenticate with HuggingFace
    # huggingface-cli login
    
    try:
        dataset = load_dataset("tarteel-ai/everyayah", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Make sure you have access to the dataset and are logged in:")
        print("  pip install huggingface_hub")
        print("  huggingface-cli login")
        return
    
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize phonemizer
    phonemizer = None
    if HAS_PHONEMIZER:
        print("Initializing Quranic Phonemizer...")
        phonemizer = Phonemizer()
    
    # Filter to Juz' Amma and process
    manifests = {"train": [], "dev": [], "test": []}
    
    processed = 0
    skipped = 0
    
    for idx, sample in enumerate(dataset):
        surah = sample.get("surah", 0)
        ayah = sample.get("ayah", 0)
        
        # Filter to Juz' Amma
        if surah not in JUZ_AMMA_SURAHS:
            skipped += 1
            continue
        
        # Get audio
        audio = sample.get("audio", {})
        audio_array = audio.get("array", None)
        sample_rate = audio.get("sampling_rate", 16000)
        
        if audio_array is None:
            skipped += 1
            continue
        
        # Save audio file
        audio_filename = f"s{surah:03d}_a{ayah:03d}_{idx:06d}.wav"
        audio_path = audio_dir / audio_filename
        sf.write(audio_path, audio_array, sample_rate)
        
        # Get phoneme transcription
        if phonemizer:
            try:
                result = phonemizer.phonemize(str(surah), stops=["verse"])
                if ayah <= len(result.verses):
                    verse = result.verses[ayah - 1]
                    phonemes = verse.phonemes_str(phoneme_sep=" ", word_sep=" ")
                else:
                    phonemes = ""
            except Exception as e:
                print(f"Warning: Failed to phonemize {surah}:{ayah}: {e}")
                phonemes = ""
        else:
            # Placeholder - you'll need to generate phonemes separately
            phonemes = sample.get("text", "")
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        # Create manifest entry
        entry = {
            "audio_filepath": str(audio_path.absolute()),
            "duration": round(duration, 3),
            "text": phonemes,
            "surah": surah,
            "ayah": ayah,
        }
        
        # Split: 90% train, 5% dev, 5% test
        # Use surah-based splitting for better evaluation
        if surah in [112, 113, 114]:  # Last 3 surahs for test
            manifests["test"].append(entry)
        elif surah in [110, 111]:  # 2 surahs for dev
            manifests["dev"].append(entry)
        else:
            manifests["train"].append(entry)
        
        processed += 1
        
        if processed % 100 == 0:
            print(f"Processed {processed} samples...")
    
    print(f"\nProcessed {processed} samples, skipped {skipped}")
    
    # Save manifests
    for split, entries in manifests.items():
        manifest_path = output_path / f"manifest_{split}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Saved {split} manifest: {len(entries)} samples -> {manifest_path}")
    
    # Save vocabulary
    vocab = extract_vocabulary(manifests)
    vocab_path = output_path / "tokens.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for idx, token in enumerate(vocab):
            f.write(f"{token} {idx}\n")
    print(f"Saved vocabulary: {len(vocab)} tokens -> {vocab_path}")
    
    print("\nData preparation complete!")


def extract_vocabulary(manifests: dict) -> list:
    """Extract unique phoneme tokens from manifests."""
    tokens = set()
    
    for split, entries in manifests.items():
        for entry in entries:
            phonemes = entry.get("text", "").split()
            tokens.update(phonemes)
    
    # Sort for consistency
    vocab = sorted(list(tokens))
    
    # Add blank token at the end (for CTC)
    vocab.append("<blank>")
    
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for processed data")
    args = parser.parse_args()
    
    prepare_data(args.output_dir)
