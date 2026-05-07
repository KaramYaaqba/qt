import { useEffect, useRef } from 'react';
import type { PageAyahInfo, LetterResult, CandidatePosition } from '../types';

const DIACRITICS_RE = /[ً-ٟؐ-ؚۖ-ۜ۟-۪ۤۧۨ-ۭ]/g;

function isDiacritic(ch: string) {
  return DIACRITICS_RE.test(ch);
}

function toArabicNumeral(n: number): string {
  return n.toString().replace(/\d/g, d => '٠١٢٣٤٥٦٧٨٩'[+d]);
}

function letterColor(r: LetterResult): string {
  if (r.status === 'error') return 'text-red-600';
  if (r.tajweed_status === 'error') return 'text-yellow-600';
  if (r.status === 'correct') return 'text-green-700';
  return '';
}

// Group a word's characters into (base letter + its diacritics) pairs
function groupLetters(text: string) {
  const groups: { letter: string; diacritics: string }[] = [];
  for (const ch of text) {
    if (isDiacritic(ch)) {
      if (groups.length > 0) groups[groups.length - 1].diacritics += ch;
    } else {
      groups.push({ letter: ch, diacritics: '' });
    }
  }
  return groups;
}

const BISMILLAH = 'بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ';

// Surahs that don't start with bismillah (At-Tawbah=9, Al-Fatiha handled differently)
const NO_BISMILLAH = new Set([9]);

interface WordProps {
  word: string;
  isActive: boolean;
  isCandidate: boolean;
  letterResults: LetterResult[] | undefined;
  charStart: number;
}

function Word({ word, isActive, isCandidate, letterResults, charStart }: WordProps) {
  let bg = '';
  if (isActive) bg = 'bg-yellow-300 rounded';
  else if (isCandidate) bg = 'bg-yellow-100 rounded opacity-60';

  if (!letterResults) {
    return <span className={`inline ${bg} px-px`}>{word}</span>;
  }

  const byPos = new Map(letterResults.map(r => [r.position, r]));
  const groups = groupLetters(word);
  let pos = charStart;

  return (
    <span className={`inline ${bg} px-px`}>
      {groups.map(({ letter, diacritics }, i) => {
        const r = byPos.get(pos++);
        return (
          <span key={i} className={r ? letterColor(r) : ''}>
            {letter}{diacritics}
          </span>
        );
      })}
    </span>
  );
}

interface SurahHeaderProps {
  nameAr: string;
  nameEn: string;
  surahNumber: number;
}

function SurahHeader({ nameAr, nameEn, surahNumber }: SurahHeaderProps) {
  return (
    <div className="my-4 mx-2">
      {/* Surah name box */}
      <div className="border-2 border-amber-700 rounded-lg py-2 px-4 text-center mb-3 bg-amber-50">
        <p className="font-quran text-amber-900 text-xl">{nameAr}</p>
        <p className="text-amber-700 text-xs mt-0.5 tracking-wide">{nameEn.toUpperCase()}</p>
      </div>
      {/* Bismillah */}
      {!NO_BISMILLAH.has(surahNumber) && surahNumber !== 1 && (
        <p className="font-quran text-center text-gray-800 text-xl mb-2">
          {BISMILLAH}
        </p>
      )}
    </div>
  );
}

interface MushafDisplayProps {
  ayahs: PageAyahInfo[];
  currentAyah: number;
  currentWord: number;
  candidates: CandidatePosition[];
  evalResults: Map<number, LetterResult[]>;
}

export default function MushafDisplay({
  ayahs,
  currentAyah,
  currentWord,
  candidates,
  evalResults,
}: MushafDisplayProps) {
  const activeRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [currentAyah, currentWord]);

  const candidateSet = new Set(candidates.map(c => `${c.ayah}:${c.wordIdx}`));

  // Group ayahs by surah to render surah headers
  const surahGroups: { surahNumber: number; nameAr: string; nameEn: string; ayahs: PageAyahInfo[] }[] = [];
  for (const ayah of ayahs) {
    const last = surahGroups[surahGroups.length - 1];
    if (!last || last.surahNumber !== ayah.surah) {
      surahGroups.push({
        surahNumber: ayah.surah,
        nameAr: ayah.surah_name_ar ?? '',
        nameEn: ayah.surah_name_en ?? '',
        ayahs: [ayah],
      });
    } else {
      last.ayahs.push(ayah);
    }
  }

  return (
    <div className="py-4 px-3 bg-amber-50 min-h-full">
      {surahGroups.map((group) => (
        <div key={group.surahNumber}>
          {group.nameAr && (
            <SurahHeader
              nameAr={group.nameAr}
              nameEn={group.nameEn}
              surahNumber={group.surahNumber}
            />
          )}

          {/* Justified Quran text — all ayahs in this surah on this page flow together */}
          <p className="mushaf-page text-gray-900 px-3">
            {group.ayahs.map((ayah) => {
              const results = evalResults.get(ayah.ayah);
              let charCursor = 0;

              return (
                <span key={`${ayah.surah}:${ayah.ayah}`}>
                  {ayah.word_list.map((word, wi) => {
                    const charStart = charCursor;
                    charCursor += word.length + 1;
                    const isActive = ayah.ayah === currentAyah && wi === currentWord;
                    const isCandidate = candidateSet.has(`${ayah.ayah}:${wi}`);

                    return (
                      <span
                        key={wi}
                        ref={isActive ? el => { activeRef.current = el; } : undefined}
                      >
                        <Word
                          word={word}
                          isActive={isActive}
                          isCandidate={isCandidate}
                          letterResults={results}
                          charStart={charStart}
                        />
                        {' '}
                      </span>
                    );
                  })}
                  {/* Ayah end marker */}
                  <span className="text-amber-700 font-quran">
                    ﴿{toArabicNumeral(ayah.ayah)}﴾
                  </span>
                  {' '}
                </span>
              );
            })}
          </p>
        </div>
      ))}
    </div>
  );
}
