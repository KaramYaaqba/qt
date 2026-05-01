"""
Recitation Check API Router

Handles audio upload and pronunciation checking.
"""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging

from ..models.schemas import RecitationCheckResponse, PhonemeError, LetterResult
from ..services.audio_processing import process_audio, validate_audio
from ..config import MAX_AUDIO_DURATION_SECONDS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["recitation"])


def get_speech_service():
    """Dependency to get speech-to-phoneme service."""
    from ..main import speech_service
    return speech_service


def get_reference_service():
    """Dependency to get phoneme reference service."""
    from ..main import reference_service
    return reference_service


def get_alignment_service():
    """Dependency to get alignment service."""
    from ..main import alignment_service
    return alignment_service


@router.post("/check", response_model=RecitationCheckResponse)
async def check_recitation(
    audio: UploadFile = File(..., description="Audio file (WebM, WAV, MP3, etc.)"),
    surah: int = Form(..., ge=78, le=114, description="Surah number (78-114)"),
    ayah: int = Form(..., ge=1, description="Ayah number"),
    speech_service=Depends(get_speech_service),
    reference_service=Depends(get_reference_service),
    alignment_service=Depends(get_alignment_service),
):
    """
    Check pronunciation of a Quran recitation.
    
    Upload an audio recording of reciting a specific ayah, and get back
    letter-by-letter and phoneme-by-phoneme analysis of pronunciation accuracy.
    
    **Supported audio formats:** WebM, WAV, MP3, OGG, M4A, AAC
    
    **Max duration:** 30 seconds
    """
    if speech_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again in a minute."
        )

    # Validate ayah exists
    if not reference_service.ayah_exists(surah, ayah):
        raise HTTPException(
            status_code=404,
            detail=f"Ayah not found: {surah}:{ayah}"
        )
    
    try:
        # Read and process audio
        audio_bytes = await audio.read()
        content_type = audio.content_type or "audio/webm"
        
        logger.info(f"Processing audio: {len(audio_bytes)} bytes, type: {content_type}")
        
        audio_array = process_audio(audio_bytes, content_type)
        logger.info(f"Audio processed: duration={len(audio_array)/16000:.2f}s, shape={audio_array.shape}, max_amplitude={abs(audio_array).max():.3f}")

        # Validate audio
        is_valid, error_msg = validate_audio(
            audio_array,
            max_duration=MAX_AUDIO_DURATION_SECONDS
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Get speech-to-phoneme prediction
        predicted_phonemes = speech_service.predict(
            audio_array,
            hint_surah=surah,
            hint_ayah=ayah
        )
        logger.info(f"Predicted {len(predicted_phonemes)} phonemes: {predicted_phonemes}")
        
        # Get reference data
        reference = reference_service.get_reference(surah, ayah)
        expected_phonemes = [p for p in reference["phoneme_list"] if p != "Q"]
        reference_text = reference["text_ar"]
        letter_phoneme_map = reference.get("letter_phoneme_map")

        logger.info(f"Expected {len(expected_phonemes)} phonemes: {expected_phonemes}")

        # Align and get errors
        alignment_result = alignment_service.align(
            predicted=predicted_phonemes,
            expected=expected_phonemes,
            reference_text=reference_text,
            letter_phoneme_map=letter_phoneme_map,
        )
        
        # Convert to response models
        phoneme_errors = [
            PhonemeError(**e) for e in alignment_result["phoneme_errors"]
        ]
        letter_results = [
            LetterResult(**lr) for lr in alignment_result["letter_results"]
        ]
        
        return RecitationCheckResponse(
            surah=surah,
            ayah=ayah,
            reference_text=reference_text,
            accuracy_phoneme=alignment_result["accuracy_phoneme"],
            accuracy_letter=alignment_result["accuracy_letter"],
            total_phonemes=alignment_result["total_phonemes"],
            total_errors=alignment_result["total_errors"],
            phoneme_errors=phoneme_errors,
            letter_results=letter_results,
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in check_recitation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
