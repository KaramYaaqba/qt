/**
 * RecordButton Component
 * 
 * Large circular button for recording audio.
 * Shows different states: idle, recording (pulsing), processing (spinner).
 */
import type { RecordingState } from '../types';

interface RecordButtonProps {
  state: RecordingState;
  duration: number;
  onStart: () => void;
  onStop: () => void;
  disabled?: boolean;
}

export function RecordButton({
  state,
  duration,
  onStart,
  onStop,
  disabled = false,
}: RecordButtonProps) {
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleClick = () => {
    if (state === 'idle') {
      onStart();
    } else if (state === 'recording') {
      onStop();
    }
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <button
        onClick={handleClick}
        disabled={disabled || state === 'processing'}
        className={`
          w-20 h-20 rounded-full flex items-center justify-center
          transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-offset-2
          ${
            state === 'recording'
              ? 'bg-red-500 hover:bg-red-600 focus:ring-red-300 animate-pulse'
              : state === 'processing'
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-primary hover:bg-blue-600 focus:ring-blue-300'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        aria-label={
          state === 'recording'
            ? 'Stop recording'
            : state === 'processing'
            ? 'Processing...'
            : 'Start recording'
        }
      >
        {state === 'processing' ? (
          // Spinner
          <svg
            className="animate-spin h-8 w-8 text-white"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        ) : state === 'recording' ? (
          // Stop icon
          <svg
            className="h-8 w-8 text-white"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          // Microphone icon
          <svg
            className="h-8 w-8 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
        )}
      </button>

      {/* Duration display */}
      <div className="text-center">
        {state === 'recording' ? (
          <span className="text-red-600 font-mono text-lg">
            {formatDuration(duration)} / 0:30
          </span>
        ) : state === 'processing' ? (
          <span className="text-gray-600">Processing...</span>
        ) : (
          <span className="text-gray-500">
            {disabled ? 'Select an ayah to start' : 'Tap to record'}
          </span>
        )}
      </div>
    </div>
  );
}
