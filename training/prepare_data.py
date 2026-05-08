#!/usr/bin/env python3
"""
Prepare Training Data for Quran Speech-to-Phoneme Model

Streams three datasets:
1. tarteel-ai/everyayah  — professional reciters (~100+ reciters, primary)
2. tarteel-ai/EA-UD      — diverse reciters with diacritised text (additional professional)
3. RetaSy/quranic_audio_dataset — non-Arabic speakers (learner audio)

Targets Juz' 29 + Juz' Amma (surahs 67–114).

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


_DIACRITICS = re.compile(
    u'[ً-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]'
)
_ALEF          = re.compile(r'[آأإٱ]')
_UTHMANIC_ALEF = re.compile(r'ـٰ')  # Tatweel+SuperscriptAlef (RetaSy) -> becomes ا
_TATWEEL       = re.compile(r'ـ')           # residual Tatweel/Kashida
_FARSI_YEH     = re.compile(r'ی')           # Farsi Yeh U+06CC -> Arabic ي


def normalize(text: str) -> str:
    text = _UTHMANIC_ALEF.sub('ا', text)  # must run before diacritics strip removes U+0670
    text = _ALEF.sub('ا', text)
    text = _DIACRITICS.sub('', text)
    text = _TATWEEL.sub('', text)
    text = _FARSI_YEH.sub('ي', text)
    return ' '.join(text.split())


def assign_split(surah: int) -> str:
    # Split by surah so entire surahs are held out — the model never sees any
    # recording of a dev/test surah during training. Splitting by ayah looks
    # correct but isn't: the same ayahs appear in all splits, so val_wer only
    # measures reciter generalisation, not content generalisation.
    #
    # Surah-level split (surah % 10):
    #   8  → test  (surahs 78, 88, 98, 108)
    #   9  → dev   (surahs 79, 89, 99, 109)
    #   else → train
    r = surah % 10
    if r == 8:
        return "test"
    if r == 9:
        return "dev"
    return "train"


def build_ayah_lookup(phonemizer) -> dict:
    """Build normalize(ayah_text) -> (surah, ayah) for target surahs.

    Collisions (two ayahs with identical normalized text) are logged and
    both entries are dropped — a match on ambiguous text could silently
    assign the wrong surah/ayah to a sample.
    """
    lookup = {}
    collisions: set[str] = set()
    for surah in sorted(TARGET_SURAHS):
        for ayah in range(1, 300):
            try:
                result = phonemizer.phonemize(f"{surah}:{ayah}")
                text = result._text
                if not text:
                    break
            except Exception:
                break
            key = normalize(text)
            if key in collisions:
                continue
            if key in lookup:
                existing = lookup.pop(key)
                collisions.add(key)
                print(
                    f"Warning: normalized text collision — "
                    f"{existing[0]}:{existing[1]} vs {surah}:{ayah}; both dropped from lookup"
                )
                continue
            lookup[key] = (surah, ayah)
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


_SUBSAMPLING_FACTOR = 4   # must match encoder.subsampling_factor in train_conformer_ctc.py
_FRAME_STRIDE_S     = 0.01  # 10ms mel frame stride

MAX_DURATION = 30.0  # seconds — enforced on actual decoded audio

def make_entry(audio_raw, sample_rate, surah, ayah, phonemes, idx, audio_dir, dataset_tag=""):
    try:
        audio_array = resample_audio(audio_raw, sample_rate)
    except RuntimeError as e:
        print(f"Error: {e}")
        return None

    duration = len(audio_array) / TARGET_SR
    if duration > MAX_DURATION:
        return None

    n_tokens  = len(phonemes.split())
    n_frames  = int(duration / (_FRAME_STRIDE_S * _SUBSAMPLING_FACTOR))
    if n_frames < n_tokens:
        # CTC requires at least one encoder frame per output token; samples that
        # violate this produce no valid alignment and corrupt the gradient.
        return None

    tag = f"{dataset_tag}_" if dataset_tag else ""
    audio_path = audio_dir / f"{tag}s{surah:03d}_a{ayah:03d}_{idx:06d}.wav"
    sf.write(audio_path, audio_array, TARGET_SR)
    return {
        "audio_filepath": str(audio_path.absolute()),
        "duration":       round(duration, 3),
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
    entry = make_entry(audio_raw, sr, surah, ayah, phonemes, idx, audio_dir, "retasy")
    return (entry, assign_split(surah)) if entry else (None, "too_long_or_ctc")


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
    entry = make_entry(audio_raw, sr, surah, ayah, phonemes, idx, audio_dir, "everyayah")
    return (entry, assign_split(surah)) if entry else (None, "too_long_or_ctc")


def process_ea_ud(sample, idx, audio_dir, phonemizer, ayah_lookup):
    """Process tarteel-ai/EA-UD — diverse reciters with diacritised transcription.

    EA-UD has no surah/ayah numbers; we match on normalised transcription text
    exactly like everyayah.  Long recordings (> 30s) are skipped because they
    cover multiple ayahs and cannot be reliably matched to a single ayah.
    """
    # EA-UD fields: audio, duration, transcription
    # Metadata duration used as a cheap pre-filter only; actual duration is
    # verified against MAX_DURATION in make_entry after decoding.
    if sample.get("duration", 0) > 35.0:
        return None, "too_long"
    match = ayah_lookup.get(normalize(sample.get("transcription", "")))
    if match is None:
        return None, "no_match"
    surah, ayah = match
    audio_raw, sr = decode_audio(sample.get("audio", {}), idx)
    if audio_raw is None:
        return None, "no_audio"
    phonemes = get_phonemes(phonemizer, surah, ayah)
    if not phonemes:
        return None, "no_phonemes"
    entry = make_entry(audio_raw, sr, surah, ayah, phonemes, idx, audio_dir, "eaud")
    return (entry, assign_split(surah)) if entry else (None, "too_long_or_ctc")


def process_tarteel(sample, idx, audio_dir, phonemizer):
    """Process tarteel-ai/tarteel-v1 — crowd-sourced diverse reciters."""
    # tarteel-v1 fields: surah_number, ayah_number, audio, recitation_id
    surah = sample.get("surah_number")
    ayah  = sample.get("ayah_number")
    if not surah or not ayah:
        return None, "no_ref"
    if surah not in TARGET_SURAHS:
        return None, "out_of_scope"
    audio_raw, sr = decode_audio(sample.get("audio", {}), idx)
    if audio_raw is None:
        return None, "no_audio"
    phonemes = get_phonemes(phonemizer, surah, ayah)
    if not phonemes:
        return None, "no_phonemes"
    entry = make_entry(audio_raw, sr, surah, ayah, phonemes, idx, audio_dir, "tarteel")
    return (entry, assign_split(surah)) if entry else (None, "too_long_or_ctc")


def _stream_dataset(name, hf_path, processor, add_entry, log_interval=500,
                    hf_splits=("train",)):
    """Stream one or more HuggingFace splits, process each sample, add to manifests.

    Returns (n_processed, n_skipped).
    """
    n_processed = n_skipped = 0
    for hf_split in hf_splits:
        try:
            ds = load_dataset(hf_path, split=hf_split, streaming=True)
            ds = ds.cast_column("audio", datasets_Audio(decode=False))
            print(f"\nLoading {name} ({hf_split})...")
            for idx, sample in enumerate(ds):
                entry, split = processor(sample, idx)
                if entry is None or not add_entry(entry, split):
                    n_skipped += 1
                    continue
                n_processed += 1
                if n_processed % log_interval == 0:
                    print(f"  {name}: {n_processed} processed (scanned {idx + 1} total)...")
        except Exception as e:
            print(f"{name} ({hf_split}) load failed: {e}")
    print(f"{name} done: {n_processed} samples")
    return n_processed, n_skipped


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
    ayah_counts: dict[tuple[int, int], int] = {}
    MAX_PER_AYAH = 40  # prevents Surah 112/113/114 from dominating training

    def add_entry(entry: dict, split: str) -> bool:
        if split == "train":
            key = (entry["surah"], entry["ayah"])
            if ayah_counts.get(key, 0) >= MAX_PER_AYAH:
                return False
            ayah_counts[key] = ayah_counts.get(key, 0) + 1
        manifests[split].append(entry)
        return True

    datasets = [
        # (name, hf_path, processor, log_interval, hf_splits)
        ("RetaSy",    "RetaSy/quranic_audio_dataset",
         lambda s, i: process_retasy(s, i, audio_dir, phonemizer, ayah_lookup),
         100, ("train",)),
        # everyayah has train/validation/test splits — load all three so we get
        # the full 100+ reciter diversity (~25k samples) instead of just 5k.
        ("everyayah", "tarteel-ai/everyayah",
         lambda s, i: process_everyayah(s, i, audio_dir, phonemizer, ayah_lookup),
         500, ("train", "validation", "test")),
        ("EA-UD",     "tarteel-ai/EA-UD",
         lambda s, i: process_ea_ud(s, i, audio_dir, phonemizer, ayah_lookup),
         500, ("train",)),
    ]

    total_processed = total_skipped = 0
    for name, hf_path, processor, log_interval, hf_splits in datasets:
        n, s = _stream_dataset(name, hf_path, processor, add_entry, log_interval, hf_splits)
        total_processed += n
        total_skipped += s

    print(f"\nTotal: {total_processed} processed, {total_skipped} skipped")

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
