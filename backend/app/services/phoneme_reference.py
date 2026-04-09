"""
Phoneme Reference Service

Loads and serves pre-generated phoneme data for Juz' Amma (Surahs 78-114).
"""
import json
from pathlib import Path
from typing import Optional


class PhonemeReferenceService:
    """
    Service for accessing pre-generated phoneme references for Quranic ayahs.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize service with path to phoneme data JSON.
        
        Args:
            data_path: Path to juz_amma_phonemes.json
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Phoneme data not found: {data_path}")
            
        with open(self.data_path, encoding="utf-8") as f:
            self.data = json.load(f)
            
        # Build surah index for faster lookups
        self._build_surah_index()
    
    def _build_surah_index(self):
        """Build an index of surahs and their metadata."""
        self.surahs = {}
        
        for key, val in self.data.items():
            sn = val["surah_number"]
            if sn not in self.surahs:
                self.surahs[sn] = {
                    "number": sn,
                    "name_ar": val["surah_name_ar"],
                    "name_en": val["surah_name_en"],
                    "ayah_count": 0,
                    "ayahs": {}
                }
            
            ayah_num = val["ayah_number"]
            self.surahs[sn]["ayahs"][ayah_num] = key
            self.surahs[sn]["ayah_count"] = max(
                self.surahs[sn]["ayah_count"], 
                ayah_num
            )

    def get_phonemes(self, surah: int, ayah: int) -> list[str]:
        """
        Get phoneme list for a specific ayah.
        
        Args:
            surah: Surah number (78-114)
            ayah: Ayah number
            
        Returns:
            List of phoneme symbols
            
        Raises:
            KeyError: If surah:ayah not found
        """
        key = f"{surah}:{ayah}"
        if key not in self.data:
            raise KeyError(f"Ayah not found: {key}")
        return self.data[key]["phoneme_list"]

    def get_text(self, surah: int, ayah: int) -> str:
        """
        Get Arabic text for a specific ayah.
        
        Args:
            surah: Surah number (78-114)
            ayah: Ayah number
            
        Returns:
            Arabic text string
            
        Raises:
            KeyError: If surah:ayah not found
        """
        key = f"{surah}:{ayah}"
        if key not in self.data:
            raise KeyError(f"Ayah not found: {key}")
        return self.data[key]["text_ar"]

    def get_reference(self, surah: int, ayah: int) -> dict:
        """
        Get full reference data for a specific ayah.
        
        Args:
            surah: Surah number (78-114)
            ayah: Ayah number
            
        Returns:
            Dictionary with all ayah data
            
        Raises:
            KeyError: If surah:ayah not found
        """
        key = f"{surah}:{ayah}"
        if key not in self.data:
            raise KeyError(f"Ayah not found: {key}")
        return self.data[key]

    def get_surah_list(self) -> list[dict]:
        """
        Get list of all surahs in Juz' Amma.
        
        Returns:
            List of surah info dictionaries sorted by number
        """
        return [
            {
                "number": s["number"],
                "name_ar": s["name_ar"],
                "name_en": s["name_en"],
                "ayah_count": s["ayah_count"],
            }
            for s in sorted(self.surahs.values(), key=lambda x: x["number"])
        ]
    
    def get_surah_info(self, surah: int) -> Optional[dict]:
        """
        Get info for a specific surah.
        
        Args:
            surah: Surah number (78-114)
            
        Returns:
            Surah info dictionary or None if not found
        """
        if surah not in self.surahs:
            return None
        s = self.surahs[surah]
        return {
            "number": s["number"],
            "name_ar": s["name_ar"],
            "name_en": s["name_en"],
            "ayah_count": s["ayah_count"],
        }
    
    def surah_exists(self, surah: int) -> bool:
        """Check if a surah exists in the data."""
        return surah in self.surahs
    
    def ayah_exists(self, surah: int, ayah: int) -> bool:
        """Check if a specific ayah exists."""
        return f"{surah}:{ayah}" in self.data
    
    def get_all_keys(self) -> list[str]:
        """Get all surah:ayah keys."""
        return list(self.data.keys())
