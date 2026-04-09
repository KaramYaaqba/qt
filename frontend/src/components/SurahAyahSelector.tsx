/**
 * SurahAyahSelector Component
 * 
 * Dropdown selectors for choosing a surah and ayah from Juz' Amma.
 */
import { useState, useEffect } from 'react';
import { getSurahs } from '../services/api';
import type { SurahInfo } from '../types';

interface SurahAyahSelectorProps {
  onSelect: (surah: number, ayah: number) => void;
  disabled?: boolean;
}

export function SurahAyahSelector({ onSelect, disabled = false }: SurahAyahSelectorProps) {
  const [surahs, setSurahs] = useState<SurahInfo[]>([]);
  const [selectedSurah, setSelectedSurah] = useState<number | null>(null);
  const [selectedAyah, setSelectedAyah] = useState<number | null>(null);
  const [ayahCount, setAyahCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load surahs on mount
  useEffect(() => {
    async function loadSurahs() {
      try {
        const data = await getSurahs();
        setSurahs(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load surahs');
        setLoading(false);
      }
    }
    loadSurahs();
  }, []);

  // Update ayah count when surah changes
  useEffect(() => {
    if (selectedSurah) {
      const surah = surahs.find((s) => s.number === selectedSurah);
      if (surah) {
        setAyahCount(surah.ayah_count);
        setSelectedAyah(1); // Reset to first ayah
        onSelect(selectedSurah, 1);
      }
    }
  }, [selectedSurah, surahs, onSelect]);

  const handleSurahChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      setSelectedSurah(value);
    }
  };

  const handleAyahChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value) && selectedSurah) {
      setSelectedAyah(value);
      onSelect(selectedSurah, value);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-4">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-100 text-red-700 rounded-lg">
        {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col sm:flex-row gap-4">
      {/* Surah Selector */}
      <div className="flex-1">
        <label htmlFor="surah-select" className="block text-sm font-medium text-gray-700 mb-1">
          Surah
        </label>
        <select
          id="surah-select"
          value={selectedSurah || ''}
          onChange={handleSurahChange}
          disabled={disabled}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
        >
          <option value="">Select a Surah</option>
          {surahs.map((surah) => (
            <option key={surah.number} value={surah.number}>
              {surah.number}. {surah.name_en} ({surah.name_ar})
            </option>
          ))}
        </select>
      </div>

      {/* Ayah Selector */}
      <div className="flex-1">
        <label htmlFor="ayah-select" className="block text-sm font-medium text-gray-700 mb-1">
          Ayah
        </label>
        <select
          id="ayah-select"
          value={selectedAyah || ''}
          onChange={handleAyahChange}
          disabled={disabled || !selectedSurah}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
        >
          <option value="">Select an Ayah</option>
          {Array.from({ length: ayahCount }, (_, i) => i + 1).map((num) => (
            <option key={num} value={num}>
              Ayah {num}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
