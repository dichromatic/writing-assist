import { describe, expect, it } from 'vitest';

import type { ParsedSpan } from './types';
import { mapSelectionToParsedSpans } from './selection';

function span(ordinal: number, startChar: number, endChar: number, text: string): ParsedSpan {
  return {
    ordinal,
    span_type: 'paragraph',
    text,
    normalized_text: text,
    start_line: ordinal,
    end_line: ordinal,
    start_byte: startChar,
    end_byte: endChar,
    start_char: startChar,
    end_char: endChar
  };
}

describe('mapSelectionToParsedSpans', () => {
  const spans = [
    span(0, 0, 11, 'First span.'),
    span(1, 13, 25, 'Second span.'),
    span(2, 27, 38, 'Third span.')
  ];

  it('returns an empty selection when the range has no text', () => {
    expect(mapSelectionToParsedSpans(spans, 'First span.', 4, 4)).toEqual({
      selectedText: '',
      startChar: 4,
      endChar: 4,
      overlappingSpanOrdinals: [],
      primarySpanOrdinal: null
    });
  });

  it('maps a single-span selection to that span', () => {
    expect(mapSelectionToParsedSpans(spans, 'First span.\n\nSecond span.', 13, 19)).toEqual({
      selectedText: 'Second',
      startChar: 13,
      endChar: 19,
      overlappingSpanOrdinals: [1],
      primarySpanOrdinal: 1
    });
  });

  it('maps a cross-span selection to all overlapping spans and keeps the first overlap primary', () => {
    expect(mapSelectionToParsedSpans(spans, 'First span.\n\nSecond span.\n\nThird span.', 8, 32)).toEqual({
      selectedText: 'an.\n\nSecond span.\n\nThird',
      startChar: 8,
      endChar: 32,
      overlappingSpanOrdinals: [0, 1, 2],
      primarySpanOrdinal: 0
    });
  });

  it('normalizes reversed selection ranges before mapping', () => {
    expect(mapSelectionToParsedSpans(spans, 'First span.\n\nSecond span.', 19, 13)).toEqual({
      selectedText: 'Second',
      startChar: 13,
      endChar: 19,
      overlappingSpanOrdinals: [1],
      primarySpanOrdinal: 1
    });
  });

  it('compares CodeMirror code-unit offsets against parser character offsets', () => {
    const markdown = 'Before 😀 emoji.\n\nAfter emoji.';
    const emojiSpans = [
      span(0, 0, 15, 'Before 😀 emoji.'),
      span(1, 17, 29, 'After emoji.')
    ];

    expect(mapSelectionToParsedSpans(emojiSpans, markdown, 18, 30)).toEqual({
      selectedText: 'After emoji.',
      startChar: 17,
      endChar: 29,
      overlappingSpanOrdinals: [1],
      primarySpanOrdinal: 1
    });
  });
});
