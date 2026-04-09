/**
 * PhonemeErrorDetail Component
 * 
 * Modal/popover showing detailed phoneme error information.
 */
import type { LetterResult } from '../types';

interface PhonemeErrorDetailProps {
  letterResult: LetterResult;
  onClose: () => void;
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

export function PhonemeErrorDetail({ letterResult, onClose }: PhonemeErrorDetailProps) {
  const errorInfo = ERROR_TYPE_LABELS[letterResult.error_type || 'replace'];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6 relative">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
          aria-label="Close"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>

        {/* Letter display */}
        <div className="text-center mb-6">
          <span
            dir="rtl"
            lang="ar"
            className="font-arabic text-6xl text-red-600"
          >
            {letterResult.letter}
          </span>
        </div>

        {/* Error type */}
        <div className="mb-4">
          <span className="inline-block px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium">
            {errorInfo.label}
          </span>
        </div>

        {/* Error description */}
        <p className="text-gray-600 mb-6">{errorInfo.description}</p>

        {/* Phoneme comparison */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          {/* Expected */}
          <div className="bg-green-50 rounded-lg p-4">
            <div className="text-sm text-green-800 font-medium mb-1">Expected</div>
            <div className="text-2xl font-mono text-green-700">
              {letterResult.expected_phoneme || '—'}
            </div>
            <div className="text-sm text-green-600 mt-1">
              {letterResult.expected_phoneme
                ? PHONEME_DESCRIPTIONS[letterResult.expected_phoneme] || letterResult.expected_phoneme
                : 'No sound expected'}
            </div>
          </div>

          {/* Detected */}
          <div className="bg-red-50 rounded-lg p-4">
            <div className="text-sm text-red-800 font-medium mb-1">Detected</div>
            <div className="text-2xl font-mono text-red-700">
              {letterResult.got_phoneme || '—'}
            </div>
            <div className="text-sm text-red-600 mt-1">
              {letterResult.got_phoneme
                ? PHONEME_DESCRIPTIONS[letterResult.got_phoneme] || letterResult.got_phoneme
                : 'No sound detected'}
            </div>
          </div>
        </div>

        {/* Tips */}
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-sm font-medium text-blue-800 mb-1">Tip</div>
          <p className="text-sm text-blue-700">
            {letterResult.expected_phoneme === 'ʕ' &&
              'The ع (ayn) comes from deep in the throat. Practice by tightening your throat muscles.'}
            {letterResult.expected_phoneme === 'ħ' &&
              'The ح (ha) is a breathy H from the throat, different from the regular ه.'}
            {letterResult.expected_phoneme === 'q' &&
              'The ق (qaf) is pronounced from the back of the throat, deeper than a regular K.'}
            {letterResult.expected_phoneme === 'ʔ' &&
              'The hamza (ء) is a glottal stop - a brief pause in your throat.'}
            {!['ʕ', 'ħ', 'q', 'ʔ'].includes(letterResult.expected_phoneme || '') &&
              'Listen to a Qari recitation and practice this sound slowly.'}
          </p>
        </div>

        {/* Close button */}
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
