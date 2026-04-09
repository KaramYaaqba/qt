/**
 * API service for Quran Recitation Checker
 */
import axios from 'axios';
import type { SurahInfo, AyahInfo, RecitationCheckResponse } from '../types';

const API_BASE = '/api';

/**
 * Get list of all surahs in Juz' Amma
 */
export async function getSurahs(): Promise<SurahInfo[]> {
  const response = await axios.get<SurahInfo[]>(`${API_BASE}/surahs`);
  return response.data;
}

/**
 * Get information about a specific surah
 */
export async function getSurah(surahNumber: number): Promise<SurahInfo> {
  const response = await axios.get<SurahInfo>(`${API_BASE}/surah/${surahNumber}`);
  return response.data;
}

/**
 * Get a specific ayah with text and phonemes
 */
export async function getAyah(surahNumber: number, ayahNumber: number): Promise<AyahInfo> {
  const response = await axios.get<AyahInfo>(
    `${API_BASE}/surah/${surahNumber}/ayah/${ayahNumber}`
  );
  return response.data;
}

/**
 * Get all ayahs of a surah
 */
export async function getSurahAyahs(surahNumber: number): Promise<AyahInfo[]> {
  const response = await axios.get<AyahInfo[]>(`${API_BASE}/surah/${surahNumber}/ayahs`);
  return response.data;
}

/**
 * Check recitation by uploading audio
 */
export async function checkRecitation(
  audioBlob: Blob,
  surah: number,
  ayah: number
): Promise<RecitationCheckResponse> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');
  formData.append('surah', surah.toString());
  formData.append('ayah', ayah.toString());

  const response = await axios.post<RecitationCheckResponse>(
    `${API_BASE}/check`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
}
