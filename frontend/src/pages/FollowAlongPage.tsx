import { useState, useCallback } from 'react';
import type { SurahPageResponse, LetterResult } from '../types';
import { getSurahPage, getSurahs } from '../services/api';
import { useFollowAlongRecorder } from '../hooks/useFollowAlongRecorder';
import MushafDisplay from '../components/MushafDisplay';
import { useEffect } from 'react';
import type { SurahInfo } from '../types';

interface Props {
  onBack: () => void;
}

export default function FollowAlongPage({ onBack }: Props) {
  const [surahs, setSurahs] = useState<SurahInfo[]>([]);
  const [selectedSurah, setSelectedSurah] = useState<number | null>(null);
  const [surahData, setSurahData] = useState<SurahPageResponse | null>(null);
  const [currentAyah, setCurrentAyah] = useState(1);
  const [currentWord, setCurrentWord] = useState(0);
  const [evalResults, setEvalResults] = useState<Map<number, LetterResult[]>>(new Map());
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getSurahs().then(setSurahs).catch(() => setError('Failed to load surahs'));
  }, []);

  const onWordUpdate = useCallback((ayah: number, word: number) => {
    setCurrentAyah(ayah);
    setCurrentWord(word);
  }, []);

  const onEvalReady = useCallback((ayah: number, letterResults: LetterResult[]) => {
    setEvalResults(prev => new Map(prev).set(ayah, letterResults));
  }, []);

  const onDone = useCallback(() => {
    // recording finished
  }, []);

  const onError = useCallback((msg: string) => {
    setError(msg);
  }, []);

  const { state, start, stop } = useFollowAlongRecorder({ onWordUpdate, onEvalReady, onDone, onError });

  const handleSurahSelect = async (surahNumber: number) => {
    setSelectedSurah(surahNumber);
    setSurahData(null);
    setEvalResults(new Map());
    setCurrentAyah(1);
    setCurrentWord(0);
    setError(null);
    setLoading(true);
    try {
      const data = await getSurahPage(surahNumber);
      setSurahData(data);
    } catch {
      setError('Failed to load surah data');
    } finally {
      setLoading(false);
    }
  };

  const handleRecite = () => {
    if (!selectedSurah) return;
    setError(null);
    setEvalResults(new Map());
    setCurrentAyah(1);
    setCurrentWord(0);
    start(selectedSurah);
  };

  const isRecording = state === 'recording' || state === 'connecting';

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white/90 backdrop-blur border-b border-gray-200 px-4 py-3 flex items-center gap-3">
        <button
          onClick={onBack}
          className="text-gray-500 hover:text-gray-800 text-sm font-medium"
        >
          ← Back
        </button>
        <span className="text-gray-300">|</span>
        <span className="font-semibold text-gray-800">Follow Along</span>

        <select
          className="ml-auto border border-gray-300 rounded-lg px-3 py-1.5 text-sm bg-white"
          value={selectedSurah ?? ''}
          onChange={(e) => handleSurahSelect(Number(e.target.value))}
          disabled={isRecording}
        >
          <option value="">Select Surah</option>
          {surahs.map((s) => (
            <option key={s.number} value={s.number}>
              {s.number}. {s.name_en} — {s.name_ar}
            </option>
          ))}
        </select>
      </div>

      {/* Surah name */}
      {surahData && (
        <div className="text-center py-4">
          <p className="font-arabic text-2xl text-gray-800">{surahData.surah_name_ar}</p>
          <p className="text-sm text-gray-500">{surahData.surah_name_en}</p>
        </div>
      )}

      {/* Main content */}
      <div className="max-w-2xl mx-auto px-4 pb-32">
        {loading && (
          <p className="text-center text-gray-400 py-12">Loading surah…</p>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-red-700 text-sm mb-4">
            {error}
          </div>
        )}

        {surahData && !loading && (
          <MushafDisplay
            ayahs={surahData.ayahs}
            currentAyah={currentAyah}
            currentWord={currentWord}
            evalResults={evalResults}
          />
        )}

        {!surahData && !loading && (
          <p className="text-center text-gray-400 py-20 text-lg">
            Select a surah to begin
          </p>
        )}
      </div>

      {/* Sticky bottom bar */}
      {surahData && (
        <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 px-4 py-4 flex items-center justify-between gap-4">
          <div className="text-sm text-gray-500">
            {isRecording && (
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                Ayah {currentAyah}
              </span>
            )}
            {state === 'done' && <span className="text-green-600">Session complete</span>}
          </div>

          {!isRecording ? (
            <button
              onClick={handleRecite}
              disabled={!surahData}
              className="ml-auto bg-blue-600 text-white px-6 py-2.5 rounded-full font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {state === 'done' ? 'Recite Again' : 'Start Reciting'}
            </button>
          ) : (
            <button
              onClick={stop}
              className="ml-auto bg-red-500 text-white px-6 py-2.5 rounded-full font-medium hover:bg-red-600 transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      )}
    </div>
  );
}
