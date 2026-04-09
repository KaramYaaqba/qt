/**
 * ResultsPanel Component
 * 
 * Displays recitation check results including accuracy and error summary.
 */
import type { RecitationCheckResponse } from '../types';

interface ResultsPanelProps {
  results: RecitationCheckResponse | null;
  onTryAgain: () => void;
  onNextAyah: () => void;
  hasNextAyah: boolean;
}

export function ResultsPanel({
  results,
  onTryAgain,
  onNextAyah,
  hasNextAyah,
}: ResultsPanelProps) {
  if (!results) {
    return null;
  }

  const getAccuracyColor = (accuracy: number): string => {
    if (accuracy >= 90) return 'text-green-600';
    if (accuracy >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getAccuracyBgColor = (accuracy: number): string => {
    if (accuracy >= 90) return 'bg-green-100';
    if (accuracy >= 70) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  const getGrade = (accuracy: number): string => {
    if (accuracy >= 95) return 'Excellent!';
    if (accuracy >= 85) return 'Very Good';
    if (accuracy >= 75) return 'Good';
    if (accuracy >= 60) return 'Needs Practice';
    return 'Keep Trying';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
      {/* Accuracy Display */}
      <div className="text-center">
        <div
          className={`inline-block ${getAccuracyBgColor(
            results.accuracy_letter
          )} rounded-full p-8`}
        >
          <div
            className={`text-5xl font-bold ${getAccuracyColor(
              results.accuracy_letter
            )}`}
          >
            {results.accuracy_letter}%
          </div>
        </div>
        <p className={`mt-2 text-lg font-medium ${getAccuracyColor(results.accuracy_letter)}`}>
          {getGrade(results.accuracy_letter)}
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 text-center">
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-2xl font-bold text-gray-800">{results.total_phonemes}</div>
          <div className="text-sm text-gray-600">Total Phonemes</div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-2xl font-bold text-red-600">{results.total_errors}</div>
          <div className="text-sm text-gray-600">Errors</div>
        </div>
      </div>

      {/* Accuracy Breakdown */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">Letter Accuracy</span>
          <span className={`font-medium ${getAccuracyColor(results.accuracy_letter)}`}>
            {results.accuracy_letter}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full ${
              results.accuracy_letter >= 90
                ? 'bg-green-500'
                : results.accuracy_letter >= 70
                ? 'bg-yellow-500'
                : 'bg-red-500'
            }`}
            style={{ width: `${results.accuracy_letter}%` }}
          />
        </div>

        <div className="flex justify-between items-center mt-4">
          <span className="text-gray-600">Phoneme Accuracy</span>
          <span className={`font-medium ${getAccuracyColor(results.accuracy_phoneme)}`}>
            {results.accuracy_phoneme}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full ${
              results.accuracy_phoneme >= 90
                ? 'bg-green-500'
                : results.accuracy_phoneme >= 70
                ? 'bg-yellow-500'
                : 'bg-red-500'
            }`}
            style={{ width: `${results.accuracy_phoneme}%` }}
          />
        </div>
      </div>

      {/* Error Summary */}
      {results.total_errors > 0 && (
        <div className="space-y-2">
          <h3 className="font-medium text-gray-800">Common Mistakes</h3>
          <p className="text-sm text-gray-600">
            Click on red letters above to see detailed error information.
          </p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={onTryAgain}
          className="flex-1 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-lg font-medium transition-colors"
        >
          Try Again
        </button>
        {hasNextAyah && (
          <button
            onClick={onNextAyah}
            className="flex-1 px-4 py-2 bg-primary hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
          >
            Next Ayah
          </button>
        )}
      </div>
    </div>
  );
}
