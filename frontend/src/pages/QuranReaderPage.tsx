import { useState, useCallback, useEffect, useRef } from 'react';
import type { SurahInfo, QuranPage, LetterResult, CandidatePosition } from '../types';
import { getSurahs, getQuranPage, getSurahStartPage } from '../services/api';
import { useFollowAlongRecorder } from '../hooks/useFollowAlongRecorder';
import MushafDisplay from '../components/MushafDisplay';

interface Props {
  onBack: () => void;
}

export default function QuranReaderPage({ onBack }: Props) {
  const [surahs, setSurahs] = useState<SurahInfo[]>([]);
  const [currentPage, setCurrentPage] = useState<number>(562);
  const [pageData, setPageData] = useState<QuranPage | null>(null);
  const [nextPageData, setNextPageData] = useState<QuranPage | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentAyah, setCurrentAyah] = useState(0);
  const [currentWord, setCurrentWord] = useState(0);
  const [candidates, setCandidates] = useState<CandidatePosition[]>([]);
  const [evalResults, setEvalResults] = useState<Map<number, LetterResult[]>>(new Map());
  const [error, setError] = useState<string | null>(null);

  const isRecordingRef = useRef(false);

  useEffect(() => {
    getSurahs().then(setSurahs).catch(() => setError('Failed to load surahs'));
  }, []);

  const loadPage = useCallback(async (pageNum: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getQuranPage(pageNum);
      setPageData(data);
      setCurrentPage(pageNum);
      setCurrentAyah(data.ayahs[0]?.ayah ?? 0);
      setCurrentWord(0);
      setCandidates([]);
      setEvalResults(new Map());
      // Prefetch next page (Quran has 604 pages max)
      if (pageNum < 604) getQuranPage(pageNum + 1).then(setNextPageData).catch(() => {});
    } catch {
      setError('Failed to load page');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPage(562);
  }, [loadPage]);

  const onWordUpdate = useCallback((ayah: number, word: number) => {
    setCurrentAyah(ayah);
    setCurrentWord(word);
    setCandidates([]);

    // Auto-advance page when user recites past the last 3 ayahs
    if (pageData) {
      const ayahs = pageData.ayahs;
      const lastAyahs = ayahs.slice(-3).map(a => a.ayah);
      if (lastAyahs.includes(ayah) && nextPageData) {
        setPageData(nextPageData);
        setNextPageData(null);
        setCurrentPage(p => {
          const next = p + 1;
          if (next < 604) getQuranPage(next + 1).then(setNextPageData).catch(() => {});
          return next;
        });
      }
    }
  }, [pageData, nextPageData]);

  const onCandidates = useCallback((c: CandidatePosition[]) => {
    setCandidates(c);
  }, []);

  const onEvalReady = useCallback((ayah: number, results: LetterResult[]) => {
    setEvalResults(prev => new Map(prev).set(ayah, results));
  }, []);

  const onDone = useCallback(() => {
    isRecordingRef.current = false;
  }, []);

  const onError = useCallback((msg: string) => {
    setError(msg);
    isRecordingRef.current = false;
  }, []);

  const { state, start, stop } = useFollowAlongRecorder({
    ayahs: pageData?.ayahs ?? [],
    onWordUpdate,
    onCandidates,
    onEvalReady,
    onDone,
    onError,
  });

  const isRecording = state === 'recording' || state === 'connecting';

  const handleSurahSelect = async (surahNumber: number) => {
    if (isRecording) stop();
    try {
      const page = await getSurahStartPage(surahNumber);
      loadPage(page);
    } catch {
      setError('Failed to get surah page');
    }
  };

  const handleRecite = () => {
    setError(null);
    setEvalResults(new Map());
    setCandidates([]);
    isRecordingRef.current = true;
    start(currentPage);
  };

  const currentSurahName = pageData?.ayahs[0]
    ? `${pageData.ayahs[0].surah_name_ar}`
    : '';

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Sticky header */}
      <div className="sticky top-0 z-10 bg-white border-b border-gray-200 px-4 py-2 flex items-center gap-2">
        <button
          onClick={onBack}
          className="text-gray-500 hover:text-gray-800 text-sm font-medium shrink-0"
        >
          ← Back
        </button>

        <button
          onClick={() => loadPage(currentPage - 1)}
          disabled={currentPage <= 1 || isRecording}
          className="text-gray-500 hover:text-gray-800 disabled:opacity-30 px-2"
        >
          ‹
        </button>

        <span className="text-sm text-gray-600 shrink-0">
          {currentPage} <span className="text-gray-400 font-arabic text-base">{currentSurahName}</span>
        </span>

        <button
          onClick={() => loadPage(currentPage + 1)}
          disabled={currentPage >= 604 || isRecording}
          className="text-gray-500 hover:text-gray-800 disabled:opacity-30 px-2"
        >
          ›
        </button>

        <select
          className="ml-auto border border-gray-200 rounded-lg px-2 py-1 text-sm bg-white max-w-[160px]"
          defaultValue=""
          onChange={(e) => e.target.value && handleSurahSelect(Number(e.target.value))}
          disabled={isRecording}
        >
          <option value="">Jump to Surah</option>
          {surahs.map((s) => (
            <option key={s.number} value={s.number}>
              {s.number}. {s.name_en}
            </option>
          ))}
        </select>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Page content */}
      <div className="flex-1 max-w-2xl mx-auto w-full pb-24">
        {loading ? (
          <p className="text-center text-gray-400 py-20">Loading page {currentPage}…</p>
        ) : pageData ? (
          <MushafDisplay
            ayahs={pageData.ayahs}
            currentAyah={currentAyah}
            currentWord={currentWord}
            candidates={candidates}
            evalResults={evalResults}
          />
        ) : null}
      </div>

      {/* Sticky footer */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 px-4 py-3 flex items-center gap-3">
        <div className="text-sm text-gray-500 flex-1">
          {isRecording && candidates.length > 0 && (
            <span className="text-yellow-600">Listening for context…</span>
          )}
          {isRecording && candidates.length === 0 && currentAyah > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              <span className="font-arabic text-base">{pageData?.ayahs.find(a => a.ayah === currentAyah)?.surah_name_ar}</span>
              {' '}ayah {currentAyah}
            </span>
          )}
          {state === 'done' && <span className="text-green-600">Session complete</span>}
        </div>

        {!isRecording ? (
          <button
            onClick={handleRecite}
            disabled={!pageData}
            className="bg-blue-600 text-white px-6 py-2 rounded-full font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {state === 'done' ? 'Recite Again' : 'Start Reciting'}
          </button>
        ) : (
          <button
            onClick={stop}
            className="bg-red-500 text-white px-6 py-2 rounded-full font-medium hover:bg-red-600 transition-colors"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
