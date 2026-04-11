import { describe, expect, it } from 'vitest';

import { buildDeterministicTaskRequest } from './requestBuilder';
import type { DocumentSelectionTarget } from '$lib/project-import/selection';

function selection(overrides: Partial<DocumentSelectionTarget> = {}): DocumentSelectionTarget {
  return {
    documentPath: 'chapters/chapter-1.md',
    selectedText: 'Selected paragraph.',
    startChar: 10,
    endChar: 29,
    overlappingSpanOrdinals: [2],
    primarySpanOrdinal: 2,
    ...overrides
  };
}

describe('buildDeterministicTaskRequest', () => {
  it('maps analysis mode to an analysis task request', () => {
    expect(buildDeterministicTaskRequest('analysis', selection())).toEqual({
      ok: true,
      request: {
        mode: 'analysis',
        task_type: 'analyze_selection',
        target: {
          document_path: 'chapters/chapter-1.md',
          selected_text: 'Selected paragraph.',
          start_char: 10,
          end_char: 29,
          anchors: [{ kind: 'span', ordinal: 2 }]
        },
        available_sources: [],
        explicitly_selected_source_paths: []
      }
    });
  });

  it('maps editing mode to a rewrite task request', () => {
    const result = buildDeterministicTaskRequest('editing', selection());

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.request.task_type).toBe('rewrite_selection');
    }
  });

  it('maps ideation mode to an ideation task request', () => {
    const result = buildDeterministicTaskRequest('ideation', selection());

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.request.task_type).toBe('ideate_selection');
    }
  });

  it('surfaces invalid selection errors before task execution', () => {
    expect(buildDeterministicTaskRequest('analysis', selection({ selectedText: '' }))).toEqual({
      ok: false,
      error: 'empty_selection',
      message: 'Select text before running a task.'
    });
  });

  it('passes available sources and active context paths into the task request', () => {
    expect(
      buildDeterministicTaskRequest('analysis', selection(), {
        availableSources: [
          {
            path: 'guides/prose.md',
            kind: { source_type: 'guide', source_kind: 'prose' },
            activation_policy: 'pinned',
            review_state: 'approved'
          },
          {
            path: 'notes/scratch.md',
            kind: { source_type: 'note' },
            activation_policy: 'explicit_only',
            review_state: 'user_authored'
          }
        ],
        activeContextPaths: ['notes/scratch.md']
      })
    ).toEqual({
      ok: true,
      request: {
        mode: 'analysis',
        task_type: 'analyze_selection',
        target: {
          document_path: 'chapters/chapter-1.md',
          selected_text: 'Selected paragraph.',
          start_char: 10,
          end_char: 29,
          anchors: [{ kind: 'span', ordinal: 2 }]
        },
        available_sources: [
          {
            path: 'guides/prose.md',
            kind: { source_type: 'guide', source_kind: 'prose' },
            activation_policy: 'pinned',
            review_state: 'approved'
          },
          {
            path: 'notes/scratch.md',
            kind: { source_type: 'note' },
            activation_policy: 'explicit_only',
            review_state: 'user_authored'
          }
        ],
        explicitly_selected_source_paths: ['notes/scratch.md']
      }
    });
  });
});
