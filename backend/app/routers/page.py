"""
Quran Page Router

Provides page-based Quran data for the mushaf reader view.
"""
from fastapi import APIRouter, HTTPException, Depends

router = APIRouter(prefix="/api", tags=["page"])


def get_page_service():
    from ..main import page_service
    return page_service


def get_reference_service():
    from ..main import reference_service
    return reference_service


@router.get("/page/{page_number}")
async def get_quran_page(
    page_number: int,
    page_svc=Depends(get_page_service),
    ref_svc=Depends(get_reference_service),
):
    if not 1 <= page_number <= 604:
        raise HTTPException(status_code=400, detail="Page number must be between 1 and 604")
    try:
        ayahs = await page_svc.get_page(page_number, ref_svc)
        return {"page_number": page_number, "ayahs": ayahs}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch page data: {e}")


@router.get("/surah/{surah_number}/startpage")
async def get_surah_start_page(
    surah_number: int,
    page_svc=Depends(get_page_service),
):
    if not 1 <= surah_number <= 114:
        raise HTTPException(status_code=400, detail="Surah number must be between 1 and 114")
    try:
        page = await page_svc.get_surah_start_page(surah_number)
        return {"surah": surah_number, "page": page}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch surah page: {e}")
