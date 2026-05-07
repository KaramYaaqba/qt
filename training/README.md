# Model Training for Quran Recitation Checker

This directory contains scripts for training the Conformer-CTC speech-to-phoneme model.

**Training should be done on a GPU instance** (e.g., RunPod A100).

## Prerequisites

1. GPU instance with at least 40GB VRAM (A100 recommended) — required by `subsampling_factor: 4`
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

#### How Training Works

The script fine-tunes [NVIDIA's pretrained Arabic FastConformer](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) for Quranic phoneme recognition using a three-stage progressive unfreezing strategy. Here is what happens end-to-end:

**Architecture**

The model is a Conformer-CTC-Large (17 layers, 512-dim, ~30M params). Audio is processed as:

```
Raw audio (16kHz)
  → Mel spectrogram (80 bands, 25ms window, 10ms stride)
  → Spec augmentation (frequency & time masking)
  → Conformer encoder (17 layers, subsampling_factor=4)
  → CTC decoder (linear projection → 71 phoneme classes)
```

The subsampling factor is set to **4** (not the default 8) so the encoder produces one output frame every ~40ms. This matters for Tajweed: a subsampling factor of 8 would produce one frame every ~80ms, potentially skipping short phonemic events like Qalqalah or short vowels entirely.

**Weight Initialization**

Rather than training from scratch, the encoder weights are transplanted from the pretrained Arabic model layer-by-layer. Only layers whose shapes match are copied; the CTC decoder head is always freshly initialized because the pretrained model used a BPE vocabulary, not phoneme labels.

**3-Stage Progressive Unfreezing**

Directly fine-tuning all 30M parameters on ~32 hours of data would cause catastrophic forgetting of the pretrained Arabic representations. Instead, training unfreezes the model gradually:

| Stage | Epochs | What is trained | Encoder LR | Decoder LR |
|-------|--------|-----------------|------------|------------|
| 1 | 0–4 | Decoder only (encoder frozen) | — | 5e-4 |
| 2 | 5–9 | Top 50% of encoder layers + decoder | 1e-5 | 1e-4 |
| 3 | 10+ | Full model (all layers) | 2e-5 | 2e-4 |

Stage 3 starts with a **1,000-step linear warmup** from 10% of the target LR. This prevents the large gradient spike that would otherwise occur the moment the frozen lower encoder layers first receive gradients.

**Why no speed augmentation?**

Most ASR training pipelines apply speed perturbation (±15%) to improve robustness. This is actively harmful for Tajweed grading: the entire Madd (elongation) rules depend on distinguishing short vowels (/a/, /i/, /u/) from their long counterparts (/aː/, /iː/, /uː/). Speed perturbation randomly stretches these, teaching the model that duration is irrelevant. Only gain augmentation (±10 dB) is applied.

**Training parameters:**
- Model: Conformer-CTC-Large (30M params, 17 layers)
- Vocabulary: 71 Quranic phoneme symbols
- Subsampling factor: 4 (doubled temporal resolution vs. default 8)
- Batch size: 20 with 2-step gradient accumulation (effective batch = 40)
- Max duration: 25s per sample
- Optimizer: AdamW, cosine annealing, weight decay 5e-4
- Early stopping: patience 30 epochs on val PER (phoneme error rate)
- Expected time: 6-8 hours on A100

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
