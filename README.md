# Juz' Amma Quran Recitation Checker

A web application that detects pronunciation errors in Quranic recitation at the letter and phoneme level, covering Juz' Amma (Surahs 78-114).

## Features

- **Speech-to-Phoneme Analysis**: Uses a Conformer-CTC model trained specifically for Quranic Arabic phonemes
- **Letter-Level Feedback**: Visual color-coded feedback on Arabic text showing correct (green), missing (red), and extra (orange) letters
- **Detailed Error Analysis**: Phoneme-by-phoneme breakdown with IPA symbols and explanations
- **Accuracy Scoring**: Overall accuracy percentage and per-letter accuracy
- **All Juz' Amma Surahs**: Full coverage of Surahs 78-114

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────────────────┐
│   Browser       │     │              Backend (FastAPI)          │
│   (React)       │────▶│                                         │
│                 │     │  ┌────────────┐  ┌──────────────────┐  │
│ • Record Audio  │     │  │ Audio Proc │──│ Speech-to-Phoneme│  │
│ • Display Text  │     │  └────────────┘  │ (ONNX Conformer) │  │
│ • Show Feedback │     │                  └──────────────────┘  │
└─────────────────┘     │                          │              │
                        │  ┌────────────┐          ▼              │
                        │  │ Reference  │──▶ Alignment Service    │
                        │  │ Phonemes   │                         │
                        │  └────────────┘                         │
                        └─────────────────────────────────────────┘
```

## Tech Stack

**Backend:**
- Python 3.11, FastAPI 0.109
- ONNX Runtime for model inference
- librosa + ffmpeg for audio processing
- python-Levenshtein for alignment

**Frontend:**
- React 18, TypeScript, Vite
- Tailwind CSS for styling
- MediaRecorder API for audio capture

**Model:**
- Conformer-CTC-Small (30M parameters)
- 71-phoneme Quranic vocabulary
- INT8 quantized ONNX

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the application (uses mock model by default)
docker compose up --build

# Access the app at http://localhost:3000
```

### Manual Development Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start server (mock mode)
USE_MOCK=true uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# Access at http://localhost:5173
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK` | `true` | Use mock speech-to-phoneme service |
| `MODEL_PATH` | `./model/model.onnx` | Path to ONNX model |
| `LOG_LEVEL` | `INFO` | Logging level |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/check` | POST | Analyze recitation audio |
| `/api/surahs` | GET | List all Juz' Amma surahs |
| `/api/surah/{n}/ayah/{m}` | GET | Get specific ayah data |
| `/health` | GET | Health check |

### POST /api/check

```bash
curl -X POST http://localhost:8000/api/check \
  -F "audio=@recitation.webm" \
  -F "surah=112" \
  -F "ayah=1"
```

Response:
```json
{
  "ayah_text": "قُلْ هُوَ اللَّهُ أَحَدٌ",
  "letters": [
    {"letter": "قُ", "phoneme": "qu", "status": "correct"},
    {"letter": "لْ", "phoneme": "l", "status": "correct"},
    ...
  ],
  "phoneme_errors": [
    {"expected": "Ɂ", "predicted": "", "type": "deletion", "position": 5}
  ],
  "accuracy": 95.2
}
```

## Training Your Own Model

See [training/README.md](training/README.md) for instructions on training the Conformer-CTC model on the EveryAyah dataset.

Quick overview:
1. `python training/prepare_data.py` - Prepare dataset
2. `python training/train_conformer_ctc.py` - Train model
3. `python training/export_onnx.py` - Export to ONNX
4. Copy `model.onnx` and `tokens.txt` to `backend/model/`
5. Set `USE_MOCK=false`

## Project Structure

```
quran-recitation-checker/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── routers/             # API routes
│   │   ├── models/              # Pydantic schemas
│   │   ├── services/            # Business logic
│   │   └── data/                # Phoneme reference data
│   ├── model/                   # ONNX model files
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/               # Page components
│   │   ├── components/          # UI components
│   │   ├── hooks/               # Custom React hooks
│   │   ├── services/            # API client
│   │   └── types/               # TypeScript types
│   ├── Dockerfile
│   └── nginx.conf
├── training/
│   ├── prepare_data.py          # Data preparation
│   ├── train_conformer_ctc.py   # Model training
│   └── export_onnx.py           # ONNX export
└── docker-compose.yml
```

## Development Notes

### Mock Mode

For development without a trained model, the app uses `MockSpeechToPhonemeService` which:
- Returns the reference phonemes with ~15% random errors
- Simulates realistic deletion, substitution, and insertion errors
- Allows full UI/UX development

### Adding New Surahs

Phoneme data is stored in `backend/app/data/juz_amma_phonemes.json`. To add more ayahs:

```json
{
  "surah_number": {
    "ayah_number": {
      "text": "Arabic text with diacritics",
      "phonemes": ["list", "of", "phonemes"]
    }
  }
}
```

Use the Quranic Phonemizer to generate phoneme transcriptions:
```bash
cd backend
git clone https://github.com/Hetchy/Quranic-Phonemizer.git phonemizer
python scripts/generate_phonemes.py
```

## License

MIT
