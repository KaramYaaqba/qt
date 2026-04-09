# Model Training for Quran Recitation Checker

This directory contains scripts for training the Conformer-CTC speech-to-phoneme model.

**Training should be done on a GPU instance** (e.g., RunPod A100).

## Prerequisites

1. GPU instance with at least 24GB VRAM (A100 recommended)
2. Python 3.10+
3. ~50GB disk space for data and checkpoints

## Setup

```bash
# Install NeMo and dependencies
pip install -r requirements-training.txt

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Training Pipeline

### Step 1: Prepare Data

Download and prepare the EveryAyah dataset:

```bash
python prepare_data.py
```

This will:
- Download `tarteel-ai/everyayah` from HuggingFace
- Filter to Juz' Amma (surahs 78-114)
- Generate phoneme labels using Quranic Phonemizer
- Create NeMo manifest files (train/dev/test)

### Step 2: Train Model

```bash
python train_conformer_ctc.py
```

Training parameters:
- Model: Conformer-CTC-Small (30M params)
- Vocabulary: 71 Quranic phoneme symbols
- Epochs: ~50 (with early stopping)
- Batch size: 16 (adjust based on VRAM)
- Expected time: 3-5 hours on A100

### Step 3: Export to ONNX

```bash
python export_onnx.py
```

This will:
- Load the best checkpoint
- Export to ONNX format
- Apply INT8 quantization
- Generate `model.onnx` and `tokens.txt`

### Step 4: Deploy

Copy the exported files to the backend:

```bash
cp model.onnx tokens.txt ../backend/model/
```

Then set `USE_MOCK=false` in your environment.

## Files

- `prepare_data.py` - Data downloading and preprocessing
- `train_conformer_ctc.py` - Model training script
- `export_onnx.py` - ONNX export and quantization
- `requirements-training.txt` - Training dependencies

## Custom Phoneme Vocabulary

The model uses a custom 71-phoneme vocabulary for Quranic Arabic:

- 28 consonant phonemes (including emphatics)
- 6 vowel phonemes (short and long)
- Special markers (pause, elongation)
- CTC blank token

See `tokens.txt` for the full vocabulary.
