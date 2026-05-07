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
- Download `tarteel-ai/everyayah`, `tarteel-ai/EA-UD`, and `RetaSy/quranic_audio_dataset`
- Filter to Juz' Amma (surahs 78-114)
- Generate phoneme labels using Quranic Phonemizer
- Create NeMo manifest files (train/dev/test)

### Step 2: Optional - Hyperparameter Tuning

Find optimal hyperparameters:

```bash
python tune_hyperparameters.py
```

This runs a grid search over key parameters and saves results to `tune_output/tuning_results.json`.

### Step 3: Train Model

```bash
python train_conformer_ctc.py
```

**Enhanced training features:**
- 3-stage progressive unfreezing (encoder freeze → partial → full)
- Strong data augmentation (speed, noise, gain, spec augment)
- Early stopping and learning rate monitoring
- Improved regularization (higher weight decay, longer warmup)

Training parameters:
- Model: Conformer-CTC-Large (30M params, 17 layers)
- Vocabulary: 71 Quranic phoneme symbols
- Batch size: 20 (increased for better gradient estimates)
- Max duration: 25s (reduced for more samples/epoch)
- Expected time: 4-6 hours on A100 with early stopping

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

- `prepare_data.py` - Data downloading and preprocessing (multiple datasets)
- `train_conformer_ctc.py` - Enhanced model training with improved augmentation and regularization
- `tune_hyperparameters.py` - Hyperparameter optimization script
- `export_onnx.py` - ONNX export and quantization
- `requirements-training.txt` - Training dependencies (consolidated)

## Custom Phoneme Vocabulary

The model uses a custom 71-phoneme vocabulary for Quranic Arabic:

- 28 consonant phonemes (including emphatics)
- 6 vowel phonemes (short and long)
- Special markers (pause, elongation)
- CTC blank token

See `tokens.txt` for the full vocabulary.
