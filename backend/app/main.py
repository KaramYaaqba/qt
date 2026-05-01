"""
Quran Recitation Checker API

FastAPI application for checking Quran pronunciation at the phoneme level.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    USE_MOCK,
    MODEL_PATH,
    TOKENS_PATH,
    PHONEME_DATA_PATH,
    CORS_ORIGINS,
    DEBUG,
)
from .routers import recitation, quran
from .services.speech_to_phoneme import create_service
from .services.phoneme_reference import PhonemeReferenceService
from .services.alignment import AlignmentService
from .models.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances (initialized at startup)
speech_service = None
reference_service = None
alignment_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Initializes services at startup and cleans up at shutdown.
    """
    global speech_service, reference_service, alignment_service
    
    logger.info("Starting Quran Recitation Checker API...")
    logger.info(f"USE_MOCK={USE_MOCK}")
    
    # Check if phoneme data exists
    if not PHONEME_DATA_PATH.exists():
        logger.error(f"Phoneme data not found at {PHONEME_DATA_PATH}")
        logger.error("Please run: python -m scripts.generate_phoneme_data")
        raise RuntimeError("Phoneme data not found. Run generate_phoneme_data.py first.")
    
    # Initialize reference service
    reference_service = PhonemeReferenceService(str(PHONEME_DATA_PATH))
    logger.info(f"Loaded reference data for {len(reference_service.get_surah_list())} surahs")
    
    # Initialize alignment service
    alignment_service = AlignmentService()
    
    # Initialize speech-to-phoneme service
    if USE_MOCK:
        logger.info("Using MOCK speech-to-phoneme service (for development)")
        speech_service = create_service(
            use_mock=True,
            reference_data_path=str(PHONEME_DATA_PATH)
        )
    else:
        # Download model if not present (blocks startup until complete)
        import os
        logger.info(f"MODEL_PATH exists: {MODEL_PATH.exists()} — path: {MODEL_PATH}")
        logger.info(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))} HF_MODEL_REPO: {os.environ.get('HF_MODEL_REPO', 'NOT SET')}")
        if not MODEL_PATH.exists():
            hf_token = os.environ.get("HF_TOKEN", "")
            hf_repo = os.environ.get("HF_MODEL_REPO", "")
            if hf_token and hf_repo:
                logger.info(f"Downloading model from {hf_repo} ...")
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=hf_repo,
                    filename="model.onnx",
                    token=hf_token,
                    local_dir=str(MODEL_PATH.parent),
                )
                logger.info("model.onnx downloaded successfully")
            else:
                logger.error("HF_TOKEN and HF_MODEL_REPO not set — cannot download model")

        if not MODEL_PATH.exists() or not TOKENS_PATH.exists():
            logger.error(f"Model files not found at {MODEL_PATH}")
            raise RuntimeError("Model files not found. Set USE_MOCK=true or provide HF_TOKEN + HF_MODEL_REPO.")

        logger.info(f"Loading ONNX model from {MODEL_PATH}")
        speech_service = create_service(
            use_mock=False,
            model_path=str(MODEL_PATH),
            tokens_path=str(TOKENS_PATH)
        )
        logger.info("ONNX model loaded successfully")
    
    logger.info("API startup complete!")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Quran Recitation Checker API...")


# Create FastAPI app
app = FastAPI(
    title="Quran Recitation Checker API",
    description="""
    API for checking Quran pronunciation at the phoneme level.
    
    This system uses a speech-to-phoneme model to detect mispronunciations
    at both the letter and phoneme level for Juz' Amma (Surahs 78-114).
    
    ## Features
    
    - **Phoneme-level detection**: Identifies exactly which sounds are mispronounced
    - **Letter-by-letter feedback**: Color-coded results for each Arabic letter
    - **Juz' Amma support**: All surahs from An-Naba (78) to An-Nas (114)
    
    ## Usage
    
    1. Get list of surahs: `GET /api/surahs`
    2. Get ayah text: `GET /api/surah/{surah}/ayah/{ayah}`
    3. Submit recording: `POST /api/check` with audio file + surah + ayah
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recitation.router)
app.include_router(quran.router)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to docs."""
    return {"message": "Quran Recitation Checker API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the service status and configuration.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        use_mock=USE_MOCK,
    )
