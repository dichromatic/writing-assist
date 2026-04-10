import { describe, expect, it } from 'vitest';

import type { ParsedSpan } from './types';
import { mapSelectionToParsedSpans, toTaskSelectionTarget } from './selection';

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

describe('toTaskSelectionTarget', () => {
  it('maps a document selection into the core SelectionTarget wire shape', () => {
    expect(
      toTaskSelectionTarget({
        documentPath: 'chapters/chapter-1.md',
        selectedText: 'Second',
        startChar: 13,
        endChar: 19,
        overlappingSpanOrdinals: [1, 2],
        primarySpanOrdinal: 1
      })
    ).toEqual({
      ok: true,
      target: {
        document_path: 'chapters/chapter-1.md',
        selected_text: 'Second',
        start_char: 13,
        end_char: 19,
        anchors: [
          { kind: 'span', ordinal: 1 },
          { kind: 'span', ordinal: 2 }
        ]
      }
    });
  });

  it('does not infer section, scene, or window anchors from span ordinals', () => {
    const result = toTaskSelectionTarget({
      documentPath: 'chapters/chapter-1.md',
      selectedText: 'Second',
      startChar: 13,
      endChar: 19,
      overlappingSpanOrdinals: [1],
      primarySpanOrdinal: 1
    });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.target.anchors).toEqual([{ kind: 'span', ordinal: 1 }]);
    }
  });

  it('rejects empty selections before task construction', () => {
    expect(
      toTaskSelectionTarget({
        documentPath: 'chapters/chapter-1.md',
        selectedText: '',
        startChar: 13,
        endChar: 13,
        overlappingSpanOrdinals: [],
        primarySpanOrdinal: null
      })
    ).toEqual({
      ok: false,
      error: 'empty_selection',
      message: 'Select text before running a task.'
    });
  });

  it('rejects reversed character ranges consistently', () => {
    expect(
      toTaskSelectionTarget({
        documentPath: 'chapters/chapter-1.md',
        selectedText: 'Second',
        startChar: 19,
        endChar: 13,
        overlappingSpanOrdinals: [1],
        primarySpanOrdinal: 1
      })
    ).toEqual({
      ok: false,
      error: 'invalid_range',
      message: 'Selection range must have startChar before endChar.'
    });
  });

  it('rejects invalid span ordinals', () => {
    expect(
      toTaskSelectionTarget({
        documentPath: 'chapters/chapter-1.md',
        selectedText: 'Second',
        startChar: 13,
        endChar: 19,
        overlappingSpanOrdinals: [-1],
        primarySpanOrdinal: null
      })
    ).toEqual({
      ok: false,
      error: 'invalid_span_ordinal',
      message: 'Selection target contains an invalid span ordinal.'
    });
  });
});
