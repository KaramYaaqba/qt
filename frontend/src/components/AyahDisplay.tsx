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

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'correct':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      case 'partial':
        return 'text-orange-500';
      case 'space':
        return '';
      default:
        return 'text-gray-800';
    }
  };

  const getStatusBg = (status: string, isHovered: boolean): string => {
    if (!isHovered) return '';
    switch (status) {
      case 'correct':
        return 'bg-green-100';
      case 'error':
        return 'bg-red-100';
      case 'partial':
        return 'bg-orange-100';
      default:
        return '';
    }
  };

  return (
    <div
      dir="rtl"
      lang="ar"
      className="font-arabic text-3xl md:text-4xl leading-loose p-6 bg-gray-50 rounded-xl"
    >
      {groups.map((group, groupIdx) => {
        const isError = group.baseStatus === 'error';
        const firstLetter = group.letters[0];
        
        if (group.baseStatus === 'space') {
          return <span key={groupIdx}> </span>;
        }

        return (
          <span
            key={groupIdx}
            className={`${getStatusColor(group.baseStatus)} ${getStatusBg(
              group.baseStatus,
              hoveredIndex === groupIdx
            )} cursor-pointer transition-colors rounded px-0.5`}
            onMouseEnter={() => setHoveredIndex(groupIdx)}
            onMouseLeave={() => setHoveredIndex(null)}
            onClick={() => {
              if (isError && onLetterClick && firstLetter) {
                onLetterClick(firstLetter);
              }
            }}
          >
            {group.letters.map((lr) => lr.letter).join('')}
          </span>
        );
      })}
    </div>
  );
}
