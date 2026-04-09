/**
 * TypeScript types for Quran Recitation Checker
 */

export interface SurahInfo {
  number: number;
  name_ar: string;
  name_en: string;
  ayah_count: number;
}

export interface AyahInfo {
  surah: number;
  ayah: number;
  surah_name_ar: string;
  surah_name_en: string;
  text_ar: string;
  phonemes: string;
  total_phonemes: number;
}

export interface LetterResult {
  letter: string;
  status: 'correct' | 'error' | 'diacritic' | 'space' | 'special' | 'unmapped';
  position: number;
  error_type?: 'replace' | 'insert' | 'delete';
  expected_phoneme?: string;
  got_phoneme?: string;
}

export interface PhonemeError {
  type: 'replace' | 'insert' | 'delete';
  position_in_expected: number;
  position_in_predicted: number;
  expected_phoneme?: string;
  got_phoneme?: string;
}

export interface RecitationCheckResponse {
  surah: number;
  ayah: number;
  reference_text: string;
  accuracy_phoneme: number;
  accuracy_letter: number;
  total_phonemes: number;
  total_errors: number;
  phoneme_errors: PhonemeError[];
  letter_results: LetterResult[];
}

export type RecordingState = 'idle' | 'recording' | 'processing';
