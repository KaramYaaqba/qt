"""
Quran Data API Router

Provides access to surah and ayah metadata.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ..models.schemas import SurahInfo, AyahInfo

router = APIRouter(prefix="/api", tags=["quran"])


def get_reference_service():
    """Dependency to get phoneme reference service."""
    from ..main import reference_service
    return reference_service


@router.get("/surahs", response_model=List[SurahInfo])
async def get_surahs(
    reference_service=Depends(get_reference_service),
):
    """
    Get list of all surahs in Juz' Amma (78-114).
    
    Returns surah number, Arabic name, English name, and ayah count.
    """
    return reference_service.get_surah_list()


@router.get("/surah/{surah_number}", response_model=SurahInfo)
async def get_surah(
    surah_number: int,
    reference_service=Depends(get_reference_service),
):
    """
    Get information about a specific surah.
    
    **Parameters:**
    - `surah_number`: Surah number (78-114)
    """
    if not 67 <= surah_number <= 114:
        raise HTTPException(
            status_code=400,
            detail="Surah number must be between 67 and 114 (Juz' 29-30)"
        )
    
    surah_info = reference_service.get_surah_info(surah_number)
    if not surah_info:
        raise HTTPException(
            status_code=404,
            detail=f"Surah {surah_number} not found"
        )
    
    return surah_info


@router.get("/surah/{surah_number}/ayah/{ayah_number}", response_model=AyahInfo)
async def get_ayah(
    surah_number: int,
    ayah_number: int,
    reference_service=Depends(get_reference_service),
):
    """
    Get a specific ayah with its Arabic text and phoneme transcription.
    
    **Parameters:**
    - `surah_number`: Surah number (78-114)
    - `ayah_number`: Ayah number within the surah
    """
    if not 67 <= surah_number <= 114:
        raise HTTPException(
            status_code=400,
            detail="Surah number must be between 67 and 114 (Juz' 29-30)"
        )
    
    if not reference_service.ayah_exists(surah_number, ayah_number):
        raise HTTPException(
            status_code=404,
            detail=f"Ayah not found: {surah_number}:{ayah_number}"
        )
    
    ref = reference_service.get_reference(surah_number, ayah_number)
    
    return AyahInfo(
        surah=surah_number,
        ayah=ayah_number,
        surah_name_ar=ref["surah_name_ar"],
        surah_name_en=ref["surah_name_en"],
        text_ar=ref["text_ar"],
        phonemes=ref["phonemes"],
        total_phonemes=ref["total_phonemes"],
    )


@router.get("/surah/{surah_number}/ayahs", response_model=List[AyahInfo])
async def get_surah_ayahs(
    surah_number: int,
    reference_service=Depends(get_reference_service),
):
    """
    Get all ayahs of a specific surah.
    
    **Parameters:**
    - `surah_number`: Surah number (78-114)
    """
    if not 67 <= surah_number <= 114:
        raise HTTPException(
            status_code=400,
            detail="Surah number must be between 67 and 114 (Juz' 29-30)"
        )
    
    surah_info = reference_service.get_surah_info(surah_number)
    if not surah_info:
        raise HTTPException(
            status_code=404,
            detail=f"Surah {surah_number} not found"
        )
    
    ayahs = []
    for ayah_num in range(1, surah_info["ayah_count"] + 1):
        ref = reference_service.get_reference(surah_number, ayah_num)
        ayahs.append(AyahInfo(
            surah=surah_number,
            ayah=ayah_num,
            surah_name_ar=ref["surah_name_ar"],
            surah_name_en=ref["surah_name_en"],
            text_ar=ref["text_ar"],
            phonemes=ref["phonemes"],
            total_phonemes=ref["total_phonemes"],
        ))
    
    return ayahs
