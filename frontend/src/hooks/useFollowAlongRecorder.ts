import { useRef, useState, useCallback } from 'react';
import type { FollowAlongState, PositionUpdate, PageAyahInfo, CandidatePosition } from '../types';
import { getReciteWebSocketUrl } from '../services/api';

interface Options {
  ayahs: PageAyahInfo[];
  onWordUpdate: (ayah: number, word: number) => void;
  onCandidates: (candidates: CandidatePosition[]) => void;
  onEvalReady: (ayah: number, letterResults: NonNullable<PositionUpdate['letter_results']>) => void;
  onDone: () => void;
  onError: (msg: string) => void;
}

function stripDiacritics(text: string): string {
  return text.replace(/[ً-ٰؐ-ؚۖ-ۭ]/g, '').trim();
}

function editDistance(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) => i);
  for (let j = 1; j <= n; j++) {
    let prev = dp[0];
    dp[0] = j;
    for (let i = 1; i <= m; i++) {
      const temp = dp[i];
      dp[i] = a[i-1] === b[j-1] ? prev : 1 + Math.min(prev, dp[i], dp[i-1]);
      prev = temp;
    }
  }
  return dp[m];
}

// Build flat word list once — stable across Chrome restarts
function buildWordIndex(ayahs: PageAyahInfo[]) {
  return ayahs.flatMap(a =>
    a.word_list.map((w, wi) => ({
      normalized: stripDiacritics(w),
      ayah: a.ayah,
      wordIdx: wi,
    }))
  );
}

interface MatchResult {
  idx: number;                      // index of last matched word in wordIndex
  candidates: CandidatePosition[];  // all equally-good matches (ambiguous)
}

// Find matches for spokenWords starting at/after fromIdx.
// Returns the single best match OR multiple candidates if ambiguous.
function findForward(
  spokenWords: string[],
  wordIndex: ReturnType<typeof buildWordIndex>,
  fromIdx: number,
  searchAhead: number,
): MatchResult | null {
  if (spokenWords.length === 0) return null;

  const limit = Math.min(fromIdx + searchAhead, wordIndex.length - spokenWords.length);
  let bestScore = Infinity;
  const matches: Array<{ idx: number; score: number }> = [];

  for (let i = fromIdx; i <= limit; i++) {
    let score = 0;
    for (let j = 0; j < spokenWords.length; j++) {
      const ref = wordIndex[i + j];
      if (!ref) { score += 4; continue; }
      score += editDistance(spokenWords[j], ref.normalized);
    }
    score /= spokenWords.length;
    if (score < bestScore) bestScore = score;
    if (score < 2) matches.push({ idx: i + spokenWords.length - 1, score });
  }

  if (matches.length === 0) return null;

  // Keep only matches within 0.3 of best score (i.e. equally good)
  const close = matches.filter(m => m.score <= bestScore + 0.3);

  if (close.length === 1) {
    return { idx: close[0].idx, candidates: [] };
  }

  // Multiple equally-good matches — return all as candidates
  const candidates = close.map(m => ({
    ayah: wordIndex[m.idx].ayah,
    wordIdx: wordIndex[m.idx].wordIdx,
  }));
  return { idx: close[0].idx, candidates };
}

export function useFollowAlongRecorder({ ayahs, onWordUpdate, onCandidates, onEvalReady, onDone, onError }: Options) {
  const [state, setState] = useState<FollowAlongState>('idle');

  const recognitionRef = useRef<InstanceType<typeof window.SpeechRecognition> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const ayahChunksRef = useRef<BlobPart[]>([]);

  // These are refs so they survive Chrome recognition restarts
  const wordIndexRef = useRef<ReturnType<typeof buildWordIndex>>([]);
  const cursorRef = useRef(0);          // current position in flat word index
  const lockedRef = useRef(false);      // true once we've locked start with high confidence
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null); // 5s reset timer
  const currentAyahTrackRef = useRef<number | null>(null); // track current ayah for page transitions
  const currentWordTrackRef = useRef<number>(0);

  // When ayahs change (page transition) — rebuild word index and reposition cursor
  const ayahsRef = useRef(ayahs);
  if (ayahsRef.current !== ayahs && lockedRef.current) {
    const newIndex = buildWordIndex(ayahs);
    // Find cursor position in new index matching current ayah/word
    if (currentAyahTrackRef.current !== null) {
      const newCursor = newIndex.findIndex(
        w => w.ayah === currentAyahTrackRef.current && w.wordIdx === currentWordTrackRef.current
      );
      cursorRef.current = newCursor >= 0 ? newCursor : 0;
    }
    wordIndexRef.current = newIndex;
    ayahsRef.current = ayahs;
  }

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;

    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    silenceTimerRef.current = null;

    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;

    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;

    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;

    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send('END');
    wsRef.current = null;

    setState('idle');
  }, []);

  const startRecognition = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'ar-SA';
    recognition.continuous = true;
    recognition.interimResults = true;
    recognitionRef.current = recognition;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const lastResult = event.results[event.results.length - 1];
      const transcript = lastResult[0].transcript;
      const confidence = lastResult[0].confidence;
      const words = stripDiacritics(transcript).split(/\s+/).filter(Boolean);
      if (words.length === 0) return;

      // Reset silence timer on every result — user is still speaking
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = setTimeout(() => {
        // 5 seconds of silence — unlock so user can reposition
        lockedRef.current = false;
        onCandidates([]);
      }, 5000);

      const matchWords = words.slice(-2);

      if (!lockedRef.current) {
        // Not locked yet — only accept high-confidence first match (≥ 0.85)
        // Chrome often returns 0 for confidence on valid results, so treat 0 as acceptable
        const isHighConfidence = confidence === 0 || confidence >= 0.85;
        if (!isHighConfidence) return;

        const result = findForward(matchWords, wordIndexRef.current, 0, wordIndexRef.current.length);
        if (!result) return;

        cursorRef.current = result.idx;
        lockedRef.current = true;

        if (result.candidates.length > 1) {
          onCandidates(result.candidates);
          return;
        }
        onCandidates([]);
      } else {
        // Locked — only search forward, never backward
        const result = findForward(matchWords, wordIndexRef.current, cursorRef.current, 8);
        if (!result) return;

        // Allow backward movement within the same ayah only (repetition style)
        // Never go back to a previous ayah
        const currentAyah = wordIndexRef.current[cursorRef.current]?.ayah;
        const resultAyah = wordIndexRef.current[result.idx]?.ayah;
        if (resultAyah !== undefined && currentAyah !== undefined && resultAyah < currentAyah) return;
        cursorRef.current = result.idx;

        if (result.candidates.length > 1) {
          onCandidates(result.candidates);
          return;
        }
        onCandidates([]);
      }

      const { ayah, wordIdx } = wordIndexRef.current[cursorRef.current];
      currentAyahTrackRef.current = ayah;
      currentWordTrackRef.current = wordIdx;
      onWordUpdate(ayah, wordIdx);
    };

    recognition.onerror = (e: SpeechRecognitionErrorEvent) => {
      if (e.error !== 'no-speech' && e.error !== 'aborted') {
        onError(`Speech recognition error: ${e.error}`);
      }
    };

    recognition.onend = () => {
      // Chrome stops recognition automatically — restart immediately
      // cursorRef and lockedRef survive because they're refs, not closure vars
      if (recognitionRef.current) {
        try { recognitionRef.current.start(); } catch { /* already started */ }
      }
    };

    recognition.start();
  }, [onWordUpdate, onCandidates, onError]);

  const start = useCallback(async (surahNumber: number) => {
    if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
      onError('Speech recognition not supported. Use Chrome or Safari.');
      return;
    }

    setState('connecting');

    // Build word index for current page
    wordIndexRef.current = buildWordIndex(ayahs);
    ayahsRef.current = ayahs;
    cursorRef.current = 0;
    lockedRef.current = false;
    currentAyahTrackRef.current = null;
    currentWordTrackRef.current = 0;

    // Start recognition immediately — doesn't need WS
    startRecognition();

    // Mic stream for evaluation
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      onError('Microphone permission denied');
      setState('idle');
      return;
    }
    streamRef.current = stream;
    setState('recording');

    // WebSocket for async evaluation
    const ws = new WebSocket(getReciteWebSocketUrl(surahNumber));
    wsRef.current = ws;

    ws.onopen = () => {
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) ayahChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        if (ayahChunksRef.current.length > 0 && ws.readyState === WebSocket.OPEN) {
          const blob = new Blob(ayahChunksRef.current, { type: 'audio/webm' });
          blob.arrayBuffer().then(buf => ws.send(buf));
          ayahChunksRef.current = [];
        }
      };

      mediaRecorder.start();
      intervalRef.current = setInterval(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          mediaRecorder.start();
        }
      }, 5000);
    };

    ws.onmessage = (e) => {
      let msg: PositionUpdate;
      try { msg = JSON.parse(e.data as string); } catch { return; }
      if (msg.type === 'position' && msg.completed_ayah && msg.letter_results && msg.ayah !== undefined) {
        onEvalReady(msg.ayah, msg.letter_results);
      } else if (msg.type === 'done') {
        setState('done');
        onDone();
      } else if (msg.type === 'error') {
        onError(msg.message ?? 'Unknown error');
      }
    };

    ws.onerror = () => console.warn('Evaluation WS error — follow-along continues');
  }, [ayahs, onWordUpdate, onEvalReady, onDone, onError, startRecognition]);

  return { state, start, stop };
}
