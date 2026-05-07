"""
Quran Page Service

Fetches page-based Quran data from api.alquran.cloud and enriches it
with local phoneme data. Results are cached in memory.
"""
import logging
import httpx

logger = logging.getLogger(__name__)

_ALQURAN_BASE = "https://api.alquran.cloud/v1"


class PageService:
    def __init__(self):
        self._page_cache: dict[int, list[dict]] = {}
        self._surah_start_page_cache: dict[int, int] = {}

    async def get_page(self, page_number: int, reference_service) -> list[dict]:
        if page_number in self._page_cache:
            return self._page_cache[page_number]

        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{_ALQURAN_BASE}/page/{page_number}/ar")
            r.raise_for_status()

        ayahs_raw = r.json()["data"]["ayahs"]
        result = []

        for a in ayahs_raw:
            sn = a["surah"]["number"]
            an = a["numberInSurah"]
            has_local = reference_service.ayah_exists(sn, an)
            ref = reference_service.get_reference(sn, an) if has_local else None

            text_ar = ref["text_ar"] if ref else a["text"]
            result.append({
                "page": page_number,
                "surah": sn,
                "ayah": an,
                "surah_name_ar": a["surah"]["name"],
                "surah_name_en": a["surah"]["englishName"],
                "text_ar": text_ar,
                "phonemes": ref["phonemes"] if ref else "",
                "total_phonemes": ref["total_phonemes"] if ref else 0,
                "word_list": text_ar.split(),
                "has_evaluation": has_local,
            })

        self._page_cache[page_number] = result
        logger.info(f"Cached page {page_number}: {len(result)} ayahs")
        return result

    async def get_surah_start_page(self, surah_number: int) -> int:
        if surah_number in self._surah_start_page_cache:
            return self._surah_start_page_cache[surah_number]

        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{_ALQURAN_BASE}/surah/{surah_number}")
            r.raise_for_status()

        page = r.json()["data"]["ayahs"][0]["page"]
        self._surah_start_page_cache[surah_number] = page
        return page
