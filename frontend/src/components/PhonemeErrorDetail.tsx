/**
 * PhonemeErrorDetail Component
 * 
 * Modal/popover showing detailed phoneme error information.
 */
import type { LetterResult } from '../types';

interface PhonemeErrorDetailProps {
  readonly letterResult: LetterResult;
  readonly onClose: () => void;
}

// IPA phoneme descriptions
const PHONEME_DESCRIPTIONS: Record<string, string> = {
  ʔ: 'Glottal stop (همزة)',
  b: 'B sound (باء)',
  t: 'T sound (تاء)',
  θ: 'Th sound like "think" (ثاء)',
  dʒ: 'J sound (جيم)',
  ħ: 'Deep H from throat (حاء)',
  x: 'Kh sound (خاء)',
  d: 'D sound (دال)',
  ð: 'Th sound like "the" (ذال)',
  r: 'Rolled R (راء)',
  z: 'Z sound (زاي)',
  s: 'S sound (سين)',
  ʃ: 'Sh sound (شين)',
  sˤ: 'Emphatic S (صاد)',
  dˤ: 'Emphatic D (ضاد)',
  tˤ: 'Emphatic T (طاء)',
  ðˤ: 'Emphatic Dh (ظاء)',
  ʕ: 'Ayn - deep throat sound (عين)',
  ʁ: 'Gh sound from throat (غين)',
  f: 'F sound (فاء)',
  q: 'Deep K from throat (قاف)',
  k: 'K sound (كاف)',
  l: 'L sound (لام)',
  m: 'M sound (ميم)',
  n: 'N sound (نون)',
  h: 'H sound (هاء)',
  w: 'W sound (واو)',
  j: 'Y sound (ياء)',
  a: 'Short "a" (فتحة)',
  aː: 'Long "aa" (ألف)',
  i: 'Short "i" (كسرة)',
  iː: 'Long "ee" (ياء)',
  u: 'Short "u" (ضمة)',
  uː: 'Long "oo" (واو)',
};

const ERROR_TYPE_LABELS: Record<string, { label: string; description: string }> = {
  replace: {
    label: 'Wrong Sound',
    description: 'You pronounced a different sound than expected.',
  },
  insert: {
    label: 'Missing Sound',
    description: 'This sound was expected but not detected in your recitation.',
  },
  delete: {
    label: 'Extra Sound',
    description: 'An unexpected sound was detected.',
  },
};

const TAJWEED_TIPS: Record<string, string> = {
  'rˤ':  'The ر (ra) here requires tafkheem — a heavy, thick sound. Keep your tongue low.',
  'lˤlˤ': 'The لله (Allah) lam requires tafkheem — a heavy L only in this word.',
  'aˤ':  'This vowel follows an emphatic consonant — keep it heavy and full.',
  'ŋ':   'This ن has ikhfa — nasalize it lightly before the following letter, do not pronounce it clearly.',
  'ñ':   'This ن has idgham — merge it into the next letter with a nasal sound.',
  'm̃':   'This م has idgham shafawi — merge it into the following م with nasalization.',
  'a:':  'This is a long vowel (madd) — hold it for 2 counts.',
  'i:':  'This is a long vowel (madd) — hold it for 2 counts.',
  'u:':  'This is a long vowel (madd) — hold it for 2 counts.',
};

const LETTER_TIPS: Record<string, string> = {
  ʕ: 'The ع (ayn) comes from deep in the throat. Practice by tightening your throat muscles.',
  ħ: 'The ح (ha) is a breathy H from the throat, different from the regular ه.',
  q: 'The ق (qaf) is pronounced from the back of the throat, deeper than a regular K.',
  ʔ: 'The hamza (ء) is a glottal stop - a brief pause in your throat.',
};

function getTip(isTajweedOnly: boolean, expectedFull: string | undefined, expectedPhoneme: string | undefined): string {
  if (isTajweedOnly) {
    return TAJWEED_TIPS[expectedFull || ''] ?? 'Focus on applying the tajweed rule for this letter.';
  }
  return LETTER_TIPS[expectedPhoneme || ''] ?? 'Listen to a Qari recitation and practice this sound slowly.';
}

export function PhonemeErrorDetail({ letterResult, onClose }: PhonemeErrorDetailProps) {
  const isTajweedOnly = letterResult.status === 'correct' && letterResult.tajweed_status === 'error';
  const errorType = isTajweedOnly ? letterResult.tajweed_error_type : letterResult.error_type;
  const errorInfo = ERROR_TYPE_LABELS[errorType || 'replace'];
  const expectedFull = letterResult.expected_phoneme_full || letterResult.expected_phoneme;
  const gotFull = letterResult.got_phoneme_full || letterResult.got_phoneme;
  const tip = getTip(isTajweedOnly, expectedFull, letterResult.expected_phoneme);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
          aria-label="Close"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Letter display */}
        <div className="text-center mb-6">
          <span dir="rtl" lang="ar" className={`font-arabic text-6xl ${isTajweedOnly ? 'text-yellow-500' : 'text-red-600'}`}>
            {letterResult.letter}
          </span>
          {isTajweedOnly && (
            <p className="text-yellow-700 text-sm mt-2 font-medium">Letter correct — Tajweed error</p>
          )}
        </div>

        {/* Error type badge */}
        <div className="mb-4">
          <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
            isTajweedOnly ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'
          }`}>
            {isTajweedOnly ? 'Tajweed Rule' : errorInfo.label}
          </span>
        </div>

        <p className="text-gray-600 mb-6">
          {isTajweedOnly
            ? 'You said the right letter but the tajweed rule was not applied correctly.'
            : errorInfo.description}
        </p>

        {/* Phoneme comparison */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-green-50 rounded-lg p-4">
            <div className="text-sm text-green-800 font-medium mb-1">Expected</div>
            <div className="text-2xl font-mono text-green-700">{expectedFull || '—'}</div>
            <div className="text-sm text-green-600 mt-1">
              {expectedFull ? PHONEME_DESCRIPTIONS[expectedFull] || expectedFull : 'No sound expected'}
            </div>
          </div>
          <div className={`rounded-lg p-4 ${isTajweedOnly ? 'bg-yellow-50' : 'bg-red-50'}`}>
            <div className={`text-sm font-medium mb-1 ${isTajweedOnly ? 'text-yellow-800' : 'text-red-800'}`}>Detected</div>
            <div className={`text-2xl font-mono ${isTajweedOnly ? 'text-yellow-700' : 'text-red-700'}`}>{gotFull || '—'}</div>
            <div className={`text-sm mt-1 ${isTajweedOnly ? 'text-yellow-600' : 'text-red-600'}`}>
              {gotFull ? PHONEME_DESCRIPTIONS[gotFull] || gotFull : 'Not detected'}
            </div>
          </div>
        </div>

        {/* Tip */}
        <div className={`rounded-lg p-4 ${isTajweedOnly ? 'bg-yellow-50' : 'bg-blue-50'}`}>
          <div className={`text-sm font-medium mb-1 ${isTajweedOnly ? 'text-yellow-800' : 'text-blue-800'}`}>Tip</div>
          <p className={`text-sm ${isTajweedOnly ? 'text-yellow-700' : 'text-blue-700'}`}>{tip}</p>
        </div>

        <button
          onClick={onClose}
          className="w-full mt-6 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-lg font-medium transition-colors"
        >
          Close
        </button>
      </div>
    </div>
  );
}
