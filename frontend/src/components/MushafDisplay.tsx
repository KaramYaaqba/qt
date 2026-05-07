import { useEffect, useRef } from 'react';
import type { PageAyahInfo, LetterResult } from '../types';

const DIACRITICS = /[ً-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]/;

function isDiacritic(ch: string) {
  return DIACRITICS.test(ch);
}

function groupLetters(text: string): { letter: string; diacritics: string }[] {
  const groups: { letter: string; diacritics: string }[] = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (isDiacritic(ch)) {
      if (groups.length > 0) groups[groups.length - 1].diacritics += ch;
    } else {
      groups.push({ letter: ch, diacritics: '' });
    }
  }
  return groups;
}

function letterColor(r: LetterResult): string {
  if (r.status === 'error') return 'text-red-600';
  if (r.tajweed_status === 'error') return 'text-yellow-600';
  if (r.status === 'correct') return 'text-green-700';
  return '';
}

interface AyahWordSpanProps {
  ayahInfo: PageAyahInfo;
  wordIndex: number;
  word: string;
  isActiveAyah: boolean;
  isActiveWord: boolean;
  letterResults?: LetterResult[];
  wordCharStart: number;
}

function AyahWordSpan({ word, isActiveWord, letterResults, wordCharStart }: AyahWordSpanProps) {
  const base = isActiveWord
    ? 'rounded px-0.5 bg-yellow-200 transition-colors duration-100'
    : 'transition-colors duration-100';

  if (!letterResults) {
    return <span className={base}>{word}</span>;
  }

  // Map letter results by position for this word's chars
  const resultsByPos = new Map<number, LetterResult>();
  for (const r of letterResults) {
    resultsByPos.set(r.position, r);
  }

  const groups = groupLetters(word);
  let charOffset = wordCharStart;

  return (
    <span className={base}>
      {groups.map(({ letter, diacritics }, gi) => {
        const pos = charOffset++;
        const r = resultsByPos.get(pos);
        const color = r ? letterColor(r) : '';
        return (
          <span key={gi} className={color}>
            {letter}{diacritics}
          </span>
        );
      })}
    </span>
  );
}

interface AyahRowProps {
  ayah: PageAyahInfo;
  isActive: boolean;
  activeWord: number;
  letterResults?: LetterResult[];
  ayahRef: (el: HTMLDivElement | null) => void;
}

function AyahRow({ ayah, isActive, activeWord, letterResults, ayahRef }: AyahRowProps) {
  let charCursor = 0;

  return (
    <div
      ref={ayahRef}
      className={`px-4 py-3 rounded-lg transition-colors duration-200 ${
        isActive ? 'bg-blue-50 ring-2 ring-blue-300' : ''
      }`}
      dir="rtl"
    >
      <span className="font-arabic leading-loose" style={{ fontSize: '1.7rem', lineHeight: '3' }}>
        {ayah.word_list.map((word, wi) => {
          const wordCharStart = charCursor;
          charCursor += word.length + 1; // +1 for space
          return (
            <span key={wi}>
              <AyahWordSpan
                ayahInfo={ayah}
                wordIndex={wi}
                word={word}
                isActiveAyah={isActive}
                isActiveWord={isActive && wi === activeWord}
                letterResults={letterResults}
                wordCharStart={wordCharStart}
              />
              {wi < ayah.word_list.length - 1 && ' '}
            </span>
          );
        })}
        {/* Ayah number marker */}
        <span className="inline-block mx-2 text-gray-400 text-base align-middle">
          ‎﴿{ayah.ayah}﴾
        </span>
      </span>
    </div>
  );
}

interface MushafDisplayProps {
  ayahs: PageAyahInfo[];
  currentAyah: number;
  currentWord: number;
  evalResults: Map<number, LetterResult[]>;
}

export default function MushafDisplay({ ayahs, currentAyah, currentWord, evalResults }: MushafDisplayProps) {
  const rowRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  useEffect(() => {
    const el = rowRefs.current.get(currentAyah);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentAyah]);

  return (
    <div className="space-y-1 py-4">
      {ayahs.map((ayah) => (
        <AyahRow
          key={`${ayah.surah}:${ayah.ayah}`}
          ayah={ayah}
          isActive={ayah.ayah === currentAyah}
          activeWord={currentWord}
          letterResults={evalResults.get(ayah.ayah)}
          ayahRef={(el) => {
            if (el) rowRefs.current.set(ayah.ayah, el);
            else rowRefs.current.delete(ayah.ayah);
          }}
        />
      ))}
    </div>
  );
}
