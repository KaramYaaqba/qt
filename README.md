# Quran Recitation Checker

Phoneme-level Quran recitation feedback for non-Arabic speakers. Record yourself reciting an ayah and get instant letter-by-letter feedback on your pronunciation.

## What it does

- Record any ayah from the last 20 surahs (95–114)
- Model listens to your audio and outputs Arabic phonemes
- Compares your phonemes against the correct Tajweed reference
- Highlights exactly which letters you got wrong in red

## Architecture

```
Browser recording (WebM/Opus)
        ↓
FastAPI backend (audio processing)
        ↓
FastConformer-CTC ONNX model (audio → phonemes)
        ↓
Levenshtein alignment (predicted vs reference phonemes)
        ↓
Letter-level colour-coded feedback
```

## Model

- Base: `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` (pretrained Arabic)
- Fine-tuned on: RetaSy (non-native speakers) + everyayah (professional reciters)
- Training data: 3,188 samples across surahs 95–114
- Training strategy: encoder frozen for first 10 epochs, then full fine-tuning
- Final val_loss: ~3.1
- Output: 55 Quranic phoneme tokens including Tajweed rules (Idgham, Ikhfaa, Qalqala)

## Supported Surahs

Surahs 95–114 (last 20 surahs of Juz' Amma):
At-Tin, Al-Alaq, Al-Qadr, Al-Bayyinah, Az-Zalzalah, Al-Adiyat, Al-Qariah, At-Takathur, Al-Asr, Al-Humazah, Al-Fil, Quraysh, Al-Maun, Al-Kawthar, Al-Kafirun, An-Nasr, Al-Masad, Al-Ikhlas, Al-Falaq, An-Nas

## Local Development

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Clone phonemizer
git clone https://github.com/Hetchy/Quranic-Phonemizer.git phonemizer

# Run with mock model (no ONNX required)
USE_MOCK=true uvicorn app.main:app --reload

# Run with real model (requires model files in backend/model/)
USE_MOCK=false uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Training

See [training/README.md](training/README.md) for the full pipeline.

```bash
cd training

# 1. Prepare data (streams RetaSy + everyayah from HuggingFace)
python prepare_data.py --output_dir ./data

# 2. Train on GPU (tested on NVIDIA A40)
python train_conformer_ctc.py --data_dir ./data --output_dir ./output

# 3. Export to ONNX + INT8 quantization
python export_onnx.py --nemo_path ./output/fastconformer_quran_6surahs.nemo \
    --output_dir ./export --verify
```

## Deployment

- **Backend**: Railway (Docker, downloads model from private HuggingFace at startup)
- **Frontend**: Vercel (static React build, free)

Environment variables for Railway:
- `HF_MODEL_REPO` — private HuggingFace repo ID (e.g. `username/quran-recitation-model`)
- `HF_TOKEN` — HuggingFace access token with read permissions
- `USE_MOCK` — `false`

## Phoneme Scoring

Beginner-friendly mode is on by default — minor Tajweed distinctions are not penalised:
- Emphatic consonants (`rˤ`, `aˤ`) treated same as plain (`r`, `a`)
- Geminates (`bb`, `ll`) treated same as single (`b`, `l`)
- Long vowels (`a:`) treated same as short (`a`)
- Nasalization variants (`ñ`, `ŋ`) treated as plain `n`

Set `BEGINNER_MODE = False` in `backend/app/services/alignment.py` for strict Tajweed scoring.
