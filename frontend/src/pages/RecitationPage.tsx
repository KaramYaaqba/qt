/**
 * RecitationPage - Main page for Quran recitation checking
 */
import { useState, useCallback, useEffect } from 'react';
import { SurahAyahSelector } from '../components/SurahAyahSelector';
import { AyahDisplay } from '../components/AyahDisplay';
import { RecordButton } from '../components/RecordButton';
import { ResultsPanel } from '../components/ResultsPanel';
import { PhonemeErrorDetail } from '../components/PhonemeErrorDetail';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { getAyah, checkRecitation, getSurah } from '../services/api';
import type { AyahInfo, RecitationCheckResponse, LetterResult } from '../types';

export function RecitationPage() {
  // Selection state
  const [selectedSurah, setSelectedSurah] = useState<number | null>(null);
  const [selectedAyah, setSelectedAyah] = useState<number | null>(null);
  const [ayahInfo, setAyahInfo] = useState<AyahInfo | null>(null);
  const [maxAyah, setMaxAyah] = useState<number>(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Results state
  const [results, setResults] = useState<RecitationCheckResponse | null>(null);
  const [selectedError, setSelectedError] = useState<LetterResult | null>(null);

  // Audio recording
  const {
    state: recordingState,
    audioBlob,
    startRecording,
    stopRecording,
    resetRecorder,
    error: recordError,
    duration,
  } = useAudioRecorder({ maxDuration: 30 });

  // Handle surah/ayah selection
  const handleSelect = useCallback(async (surah: number, ayah: number) => {
    setSelectedSurah(surah);
    setSelectedAyah(ayah);
    setResults(null);
    setError(null);
    setSelectedError(null);
    resetRecorder();

    try {
      setLoading(true);
      const [ayahData, surahData] = await Promise.all([
        getAyah(surah, ayah),
        getSurah(surah),
      ]);
      setAyahInfo(ayahData);
      setMaxAyah(surahData.ayah_count);
    } catch (err) {
      setError('Failed to load ayah');
      setAyahInfo(null);
    } finally {
      setLoading(false);
    }
  }, [resetRecorder]);

  // Process audio when recording completes
  useEffect(() => {
    if (audioBlob && selectedSurah && selectedAyah) {
      processRecording(audioBlob);
    }
  }, [audioBlob]);

  const processRecording = async (blob: Blob) => {
    if (!selectedSurah || !selectedAyah) return;

    try {
      setError(null);
      const result = await checkRecitation(blob, selectedSurah, selectedAyah);
      setResults(result);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to check recitation');
      }
    }
  };

  const handleTryAgain = () => {
    setResults(null);
  };

  const handleNextAyah = () => {
    if (selectedSurah && selectedAyah && selectedAyah < maxAyah) {
      handleSelect(selectedSurah, selectedAyah + 1);
    }
  };

  const handleLetterClick = (letter: LetterResult) => {
    if (letter.status === 'error') {
      setSelectedError(letter);
    }
  };

  const isProcessing = recordingState === 'processing' || (!!audioBlob && !results && !error);

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <h1 className="text-2xl font-bold text-gray-900">
            Quran Recitation Checker
          </h1>
          <p className="text-gray-600 mt-1">
            Practice your Tajweed with phoneme-level feedback
          </p>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        {/* Surah/Ayah Selector */}
        <section className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Select Surah & Ayah
          </h2>
          <SurahAyahSelector
            onSelect={handleSelect}
            disabled={recordingState === 'recording' || isProcessing}
          />
        </section>

        {/* Ayah Display */}
        {loading && (
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading ayah...</p>
          </div>
        )}

        {ayahInfo && !loading && (
          <section className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-800">
                {ayahInfo.surah_name_en} ({ayahInfo.surah_name_ar}) - Ayah {ayahInfo.ayah}
              </h2>
              <span className="text-sm text-gray-500">
                {ayahInfo.total_phonemes} phonemes
              </span>
            </div>
            <AyahDisplay
              text={ayahInfo.text_ar}
              letterResults={results?.letter_results}
              onLetterClick={handleLetterClick}
            />
          </section>
        )}

        {/* Record Button */}
        {ayahInfo && !loading && (
          <section className="flex justify-center">
            <RecordButton
              state={isProcessing ? 'processing' : recordingState}
              duration={duration}
              onStart={startRecording}
              onStop={stopRecording}
              disabled={!ayahInfo}
            />
          </section>
        )}

        {/* Errors */}
        {(error || recordError) && (
          <div className="bg-red-100 border border-red-300 text-red-700 rounded-xl p-4">
            {error || recordError}
          </div>
        )}

        {/* Results Panel */}
        {results && (
          <ResultsPanel
            results={results}
            onTryAgain={handleTryAgain}
            onNextAyah={handleNextAyah}
            hasNextAyah={selectedAyah !== null && selectedAyah < maxAyah}
          />
        )}

        {/* Phoneme Error Detail Modal */}
        {selectedError && (
          <PhonemeErrorDetail
            letterResult={selectedError}
            onClose={() => setSelectedError(null)}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t mt-16">
        <div className="max-w-4xl mx-auto px-4 py-6 text-center text-gray-600 text-sm">
          <p>Juz' Amma Recitation Checker • Surahs 78-114</p>
          <p className="mt-1">
            Uses phoneme-level detection for accurate pronunciation feedback
          </p>
        </div>
      </footer>
    </div>
  );
}
