"""
API Request/Response Schemas

Pydantic models for the Quran Recitation Checker API.
"""
from pydantic import BaseModel, Field
from typing import Optional


class LetterResult(BaseModel):
    """Result for a single letter in the ayah."""
    letter: str = Field(..., description="The Arabic letter or character")
    status: str = Field(..., description="Status: 'correct', 'error', 'diacritic', 'space', 'special', 'unmapped'")
    position: int = Field(..., description="Character position in the ayah text")
    error_type: Optional[str] = Field(None, description="Type of error: 'replace', 'insert', 'delete'")
    expected_phoneme: Optional[str] = Field(None, description="Expected phoneme (IPA)")
    got_phoneme: Optional[str] = Field(None, description="Detected phoneme (IPA)")


class PhonemeError(BaseModel):
    """Details of a phoneme-level error."""
    type: str = Field(..., description="Error type: 'replace', 'insert', 'delete'")
    position_in_expected: int = Field(..., description="Position in expected phoneme sequence")
    position_in_predicted: int = Field(..., description="Position in predicted phoneme sequence")
    expected_phoneme: Optional[str] = Field(None, description="Expected phoneme (IPA)")
    got_phoneme: Optional[str] = Field(None, description="Detected phoneme (IPA)")


class RecitationCheckRequest(BaseModel):
    """Request body for checking recitation (used with form data)."""
    surah: int = Field(..., ge=78, le=114, description="Surah number (78-114 for Juz' Amma)")
    ayah: int = Field(..., ge=1, description="Ayah number")


class RecitationCheckResponse(BaseModel):
    """Response from recitation check endpoint."""
    surah: int = Field(..., description="Surah number")
    ayah: int = Field(..., description="Ayah number")
    reference_text: str = Field(..., description="Arabic text of the ayah")
    accuracy_phoneme: float = Field(..., description="Phoneme-level accuracy percentage (0-100)")
    accuracy_letter: float = Field(..., description="Letter-level accuracy percentage (0-100)")
    total_phonemes: int = Field(..., description="Total expected phonemes")
    total_errors: int = Field(..., description="Number of phoneme errors detected")
    phoneme_errors: list[PhonemeError] = Field(..., description="List of phoneme-level errors")
    letter_results: list[LetterResult] = Field(..., description="Per-letter results with status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "surah": 112,
                "ayah": 1,
                "reference_text": "قُلْ هُوَ اللَّهُ أَحَدٌ",
                "accuracy_phoneme": 85.7,
                "accuracy_letter": 88.9,
                "total_phonemes": 14,
                "total_errors": 2,
                "phoneme_errors": [
                    {
                        "type": "replace",
                        "position_in_expected": 5,
                        "position_in_predicted": 5,
                        "expected_phoneme": "ʔ",
                        "got_phoneme": "a"
                    }
                ],
                "letter_results": [
                    {"letter": "ق", "status": "correct", "position": 0, "expected_phoneme": "q"},
                    {"letter": "ُ", "status": "diacritic", "position": 1},
                ]
            }
        }


class SurahInfo(BaseModel):
    """Information about a surah."""
    number: int = Field(..., description="Surah number")
    name_ar: str = Field(..., description="Arabic name")
    name_en: str = Field(..., description="English transliteration")
    ayah_count: int = Field(..., description="Number of ayahs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "number": 112,
                "name_ar": "الإخلاص",
                "name_en": "Al-Ikhlas",
                "ayah_count": 4
            }
        }


class AyahInfo(BaseModel):
    """Information about a specific ayah."""
    surah: int = Field(..., description="Surah number")
    ayah: int = Field(..., description="Ayah number")
    surah_name_ar: str = Field(..., description="Arabic surah name")
    surah_name_en: str = Field(..., description="English surah name")
    text_ar: str = Field(..., description="Arabic text with diacritics")
    phonemes: str = Field(..., description="Phoneme transcription (space-separated)")
    total_phonemes: int = Field(..., description="Number of phonemes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "surah": 112,
                "ayah": 1,
                "surah_name_ar": "الإخلاص",
                "surah_name_en": "Al-Ikhlas",
                "text_ar": "قُلْ هُوَ اللَّهُ أَحَدٌ",
                "phonemes": "q u l | h u w a | ʔ a l l aː h u | ʔ a ħ a d u n",
                "total_phonemes": 14
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Ayah not found: 112:5"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    use_mock: bool = Field(..., description="Whether mock service is being used")
