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

export type TaskTargetAnchorKind = 'span' | 'section' | 'scene' | 'window';

export type TaskTargetAnchor = {
  kind: TaskTargetAnchorKind;
  ordinal: number;
};

// Snake_case shape mirrors the serde contract for `writing_assist_core::SelectionTarget`.
export type TaskSelectionTarget = {
  document_path: string;
  selected_text: string;
  start_char: number;
  end_char: number;
  anchors: TaskTargetAnchor[];
};

export type SelectionTargetAdapterError =
  | 'missing_document_path'
  | 'empty_selection'
  | 'invalid_range'
  | 'invalid_span_ordinal';

export type SelectionTargetAdapterResult =
  | { ok: true; target: TaskSelectionTarget }
  | { ok: false; error: SelectionTargetAdapterError; message: string };

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

export function toTaskSelectionTarget(
  selection: DocumentSelectionTarget
): SelectionTargetAdapterResult {
  if (!selection.documentPath.trim()) {
    return {
      ok: false,
      error: 'missing_document_path',
      message: 'Selection target is missing a document path.'
    };
  }

  if (!selection.selectedText) {
    return {
      ok: false,
      error: 'empty_selection',
      message: 'Select text before running a task.'
    };
  }

  if (
    !Number.isInteger(selection.startChar) ||
    !Number.isInteger(selection.endChar) ||
    selection.startChar < 0 ||
    selection.endChar <= selection.startChar
  ) {
    return {
      ok: false,
      error: 'invalid_range',
      message: 'Selection range must have startChar before endChar.'
    };
  }

  if (
    selection.overlappingSpanOrdinals.some(
      (ordinal) => !Number.isInteger(ordinal) || ordinal < 0
    )
  ) {
    return {
      ok: false,
      error: 'invalid_span_ordinal',
      message: 'Selection target contains an invalid span ordinal.'
    };
  }

  return {
    ok: true,
    target: {
      document_path: selection.documentPath,
      selected_text: selection.selectedText,
      start_char: selection.startChar,
      end_char: selection.endChar,
      // Only span anchors are available from Phase 1.8 selection state; broader targets are added later.
      anchors: selection.overlappingSpanOrdinals.map((ordinal) => ({ kind: 'span', ordinal }))
    }
  };
}
