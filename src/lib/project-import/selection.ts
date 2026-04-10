import type { ParsedSpan } from './types';

export type ParsedSelection = {
  selectedText: string;
  startChar: number;
  endChar: number;
  overlappingSpanOrdinals: number[];
  primarySpanOrdinal: number | null;
};

export type DocumentSelectionTarget = ParsedSelection & {
  documentPath: string;
};

function overlapsSelection(span: ParsedSpan, startChar: number, endChar: number): boolean {
  return span.start_char < endChar && span.end_char > startChar;
}

function codeUnitOffsetToCharOffset(markdown: string, codeUnitOffset: number): number {
  return Array.from(markdown.slice(0, codeUnitOffset)).length;
}

export function mapSelectionToParsedSpans(
  spans: ParsedSpan[],
  markdown: string,
  anchorCodeUnit: number,
  headCodeUnit: number
): ParsedSelection {
  const startCodeUnit = Math.min(anchorCodeUnit, headCodeUnit);
  const endCodeUnit = Math.max(anchorCodeUnit, headCodeUnit);
  const startChar = codeUnitOffsetToCharOffset(markdown, startCodeUnit);
  const endChar = codeUnitOffsetToCharOffset(markdown, endCodeUnit);
  const selectedText = markdown.slice(startCodeUnit, endCodeUnit);

  if (startChar === endChar) {
    return {
      selectedText,
      startChar,
      endChar,
      overlappingSpanOrdinals: [],
      primarySpanOrdinal: null
    };
  }

  // Phase 1.8 maps CodeMirror character ranges onto parser spans so Phase 2 can target selected text.
  const overlappingSpanOrdinals = spans
    .filter((span) => overlapsSelection(span, startChar, endChar))
    .map((span) => span.ordinal);

  return {
    selectedText,
    startChar,
    endChar,
    overlappingSpanOrdinals,
    primarySpanOrdinal: overlappingSpanOrdinals[0] ?? null
  };
}
