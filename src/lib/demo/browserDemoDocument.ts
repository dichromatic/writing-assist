import type {
  LoadedDocument,
  OpenedProject,
  ParsedMarkdownDocument,
  ParsedSpan
} from '$lib/project-import/types';

const demoMarkdown = `# Browser Demo

This document is embedded so the browser-only dev server can exercise CodeMirror without Tauri filesystem access.

---

Select this paragraph to verify selection targeting. Emoji 😀 is included to exercise offset conversion.`;

function firstIndex(text: string, target: string): number {
  const index = text.indexOf(target);

  if (index < 0) {
    throw new Error(`Demo fixture is missing expected text: ${target}`);
  }

  return Array.from(text.slice(0, index)).length;
}

function charLength(text: string): number {
  return Array.from(text).length;
}

function span(ordinal: number, spanType: ParsedSpan['span_type'], text: string): ParsedSpan {
  const startChar = firstIndex(demoMarkdown, text);
  const endChar = startChar + charLength(text);

  return {
    ordinal,
    span_type: spanType,
    text,
    normalized_text: text.split(/\s+/).join(' '),
    start_line: ordinal,
    end_line: ordinal,
    start_byte: startChar,
    end_byte: endChar,
    start_char: startChar,
    end_char: endChar
  };
}

const spans = [
  span(0, 'heading', '# Browser Demo'),
  span(
    1,
    'paragraph',
    'This document is embedded so the browser-only dev server can exercise CodeMirror without Tauri filesystem access.'
  ),
  span(2, 'scene', '---'),
  span(
    3,
    'paragraph',
    'Select this paragraph to verify selection targeting. Emoji 😀 is included to exercise offset conversion.'
  )
];

const parsed: ParsedMarkdownDocument = {
  spans,
  sections: [
    {
      ordinal: 0,
      text: '# Browser Demo\n\nThis document is embedded so the browser-only dev server can exercise CodeMirror without Tauri filesystem access.',
      normalized_text:
        '# Browser Demo This document is embedded so the browser-only dev server can exercise CodeMirror without Tauri filesystem access.',
      boundary_kind: 'file_start',
      boundary_text: null,
      start_line: 0,
      end_line: 2,
      start_byte: 0,
      end_byte: spans[1].end_byte,
      start_char: 0,
      end_char: spans[1].end_char
    },
    {
      ordinal: 1,
      text: spans[3].text,
      normalized_text: spans[3].normalized_text,
      boundary_kind: 'scene_break',
      boundary_text: '---',
      start_line: 6,
      end_line: 6,
      start_byte: spans[3].start_byte,
      end_byte: spans[3].end_byte,
      start_char: spans[3].start_char,
      end_char: spans[3].end_char
    }
  ],
  scenes: [
    {
      ordinal: 0,
      text: '# Browser Demo\n\nThis document is embedded so the browser-only dev server can exercise CodeMirror without Tauri filesystem access.',
      normalized_text:
        '# Browser Demo This document is embedded so the browser-only dev server can exercise CodeMirror without Tauri filesystem access.',
      separator: null,
      start_line: 0,
      end_line: 2,
      start_byte: 0,
      end_byte: spans[1].end_byte,
      start_char: 0,
      end_char: spans[1].end_char,
      start_span_ordinal: 0,
      end_span_ordinal: 1
    },
    {
      ordinal: 1,
      text: spans[3].text,
      normalized_text: spans[3].normalized_text,
      separator: '---',
      start_line: 6,
      end_line: 6,
      start_byte: spans[3].start_byte,
      end_byte: spans[3].end_byte,
      start_char: spans[3].start_char,
      end_char: spans[3].end_char,
      start_span_ordinal: 3,
      end_span_ordinal: 3
    }
  ]
};

export const browserDemoDocument: LoadedDocument = {
  document: {
    path: 'demo/browser-only.md',
    document_type: 'manuscript'
  },
  markdown: demoMarkdown,
  parsed
};

// The browser demo needs a small synthetic project so Phase 3 context-source selection can be
// exercised without Tauri filesystem access or a persisted import configuration.
export const browserDemoProject: OpenedProject = {
  config: {
    root_path: '/browser-demo',
    directory_mappings: [
      {
        path: 'demo',
        role: 'primary_manuscript',
        enabled: true
      },
      {
        path: 'guides',
        role: 'reference',
        enabled: true
      },
      {
        path: 'notes',
        role: 'notes',
        enabled: true
      }
    ]
  },
  documents: [
    {
      path: 'demo/browser-only.md',
      document_type: 'manuscript'
    },
    {
      path: 'guides/Prose Guideline.md',
      document_type: 'reference'
    },
    {
      path: 'guides/World Summary.md',
      document_type: 'reference'
    },
    {
      path: 'notes/Loose Brainstorm.md',
      document_type: 'note'
    }
  ]
};
