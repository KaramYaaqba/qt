/**
 * AyahDisplay Component
 * 
 * Displays Arabic ayah text with color-coded letter results.
 * Green = correct, Red = error, inherits for diacritics.
 */
import { useState } from 'react';
import type { LetterResult } from '../types';

interface AyahDisplayProps {
  text: string;
  letterResults?: LetterResult[];
  onLetterClick?: (letter: LetterResult) => void;
}

export function AyahDisplay({ text, letterResults, onLetterClick }: AyahDisplayProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // If no results yet, just display the text
  if (!letterResults || letterResults.length === 0) {
    return (
      <div
        dir="rtl"
        lang="ar"
        className="font-arabic text-3xl md:text-4xl leading-loose text-gray-800 p-6 bg-gray-50 rounded-xl"
      >
        {text}
      </div>
    );
  }

  // Group letters with their following diacritics
  const groups: { letters: LetterResult[]; baseStatus: string }[] = [];
  let currentGroup: LetterResult[] = [];
  let baseStatus = 'correct';

  letterResults.forEach((lr) => {
    if (lr.status === 'diacritic' || lr.status === 'special') {
      // Diacritics follow the previous base letter
      currentGroup.push(lr);
    } else if (lr.status === 'space') {
      // Flush current group
      if (currentGroup.length > 0) {
        groups.push({ letters: [...currentGroup], baseStatus });
        currentGroup = [];
      }
      // Add space as its own group
      groups.push({ letters: [lr], baseStatus: 'space' });
      baseStatus = 'correct';
    } else {
      // New base letter - flush previous group
      if (currentGroup.length > 0) {
        groups.push({ letters: [...currentGroup], baseStatus });
      }
      currentGroup = [lr];
      baseStatus = lr.status;
    }
  });

  // Don't forget the last group
  if (currentGroup.length > 0) {
    groups.push({ letters: [...currentGroup], baseStatus });
  }

  // Derive display status from letter + tajweed status:
  // 'error'         → red   (wrong letter)
  // 'tajweed_error' → yellow (correct letter, wrong tajweed)
  // 'correct'       → green
  const getDisplayStatus = (group: { letters: LetterResult[]; baseStatus: string }): string => {
    if (group.baseStatus === 'error') return 'error';
    if (group.baseStatus === 'correct') {
      const base = group.letters.find(lr => lr.status === 'correct');
      if (base?.tajweed_status === 'error') return 'tajweed_error';
    }
    return group.baseStatus;
  };

  const getStatusColor = (displayStatus: string): string => {
    switch (displayStatus) {
      case 'correct':       return 'text-green-600';
      case 'tajweed_error': return 'text-yellow-500';
      case 'error':         return 'text-red-600';
      case 'space':         return '';
      default:              return 'text-gray-800';
    }
  };

  const getStatusBg = (displayStatus: string, isHovered: boolean): string => {
    if (!isHovered) return '';
    switch (displayStatus) {
      case 'correct':       return 'bg-green-100';
      case 'tajweed_error': return 'bg-yellow-100';
      case 'error':         return 'bg-red-100';
      default:              return '';
    }
  };

  return (
    <div
      dir="rtl"
      lang="ar"
      className="font-arabic text-3xl md:text-4xl leading-loose p-6 bg-gray-50 rounded-xl"
    >
      {groups.map((group, groupIdx) => {
        const displayStatus = getDisplayStatus(group);
        const firstLetter = group.letters[0];

        const key = `g-${firstLetter?.position ?? groupIdx}`;

        if (displayStatus === 'space') {
          return <span key={key}> </span>;
        }

        const isClickable = (displayStatus === 'error' || displayStatus === 'tajweed_error') && !!onLetterClick;
        const colorClass = `${getStatusColor(displayStatus)} ${getStatusBg(displayStatus, hoveredIndex === groupIdx)} transition-colors rounded px-0.5`;
        const text = group.letters.map((lr) => lr.letter).join('');

        if (isClickable) {
          return (
            <button
              key={key}
              type="button"
              className={`font-arabic inline ${colorClass} cursor-pointer bg-transparent border-0 p-0 m-0 leading-none`}
              onMouseEnter={() => setHoveredIndex(groupIdx)}
              onMouseLeave={() => setHoveredIndex(null)}
              onClick={() => onLetterClick(firstLetter)}
            >
              {text}
            </button>
          );
        }

        return (
          <span
            key={key}
            className={colorClass}
            onMouseEnter={() => setHoveredIndex(groupIdx)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            {text}
          </span>
        );
      })}
    </div>
  );
}
