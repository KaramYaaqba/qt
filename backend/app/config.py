"""Application configuration settings."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = APP_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

# Model settings
MODEL_PATH = MODEL_DIR / "model.onnx"
TOKENS_PATH = MODEL_DIR / "tokens.txt"

# Data files
PHONEME_DATA_PATH = DATA_DIR / "juz_amma_phonemes.json"
METADATA_PATH = DATA_DIR / "juz_amma_metadata.json"
PHONEME_LETTER_MAP_PATH = DATA_DIR / "phoneme_to_letter_map.json"

# Runtime settings
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Audio settings
MAX_AUDIO_DURATION_SECONDS = 30
SAMPLE_RATE = 16000

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
