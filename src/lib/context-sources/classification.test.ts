import { describe, expect, it } from 'vitest';

import {
  buildProjectContextSources,
  classifyProjectContextSource
} from './classification';
import type { ProjectDocumentEntry } from '$lib/project-import/types';
import type { ContextSourceKind } from '$lib/task/types';

function document(
  path: string,
  documentType: ProjectDocumentEntry['document_type']
): ProjectDocumentEntry {
  return {
    path,
    document_type: documentType
  };
}

describe('project context source classification', () => {
  it('uses an explicit kind override before filename heuristics', () => {
    const explicitKind: ContextSourceKind = {
      source_type: 'guide',
      source_kind: 'rewrite'
    };

    expect(classifyProjectContextSource(document('chapters/chapter-1.md', 'manuscript'), explicitKind)).toEqual({
      path: 'chapters/chapter-1.md',
      kind: explicitKind,
      activation_policy: 'explicit_only',
      review_state: 'user_authored'
    });
  });

  it('classifies confidently named guide and reference documents', () => {
    expect(classifyProjectContextSource(document('guides/Prose Guideline.md', 'reference'))).toEqual({
      path: 'guides/Prose Guideline.md',
      kind: { source_type: 'guide', source_kind: 'prose' },
      activation_policy: 'explicit_only',
      review_state: 'user_authored'
    });

    expect(
      classifyProjectContextSource(document('reference/World Summary.md', 'reference'))
    ).toEqual({
      path: 'reference/World Summary.md',
      kind: { source_type: 'reference', source_kind: 'world_summary' },
      activation_policy: 'explicit_only',
      review_state: 'user_authored'
    });
  });

  it('classifies note documents without guessing a richer source type', () => {
    expect(classifyProjectContextSource(document('notes/scratchpad.md', 'note'))).toEqual({
      path: 'notes/scratchpad.md',
      kind: { source_type: 'note' },
      activation_policy: 'explicit_only',
      review_state: 'user_authored'
    });
  });

  it('leaves ambiguous reference documents unclassified', () => {
    expect(classifyProjectContextSource(document('reference/misc-notes.md', 'reference'))).toBeNull();
  });

  it('builds a stable ordered list of only classified project sources', () => {
    expect(
      buildProjectContextSources([
        document('chapters/chapter-1.md', 'manuscript'),
        document('guides/Style Guide.md', 'reference'),
        document('reference/World Overview.md', 'reference'),
        document('reference/loose-file.md', 'reference'),
        document('notes/brainstorm.md', 'note')
      ])
    ).toEqual([
      {
        path: 'guides/Style Guide.md',
        kind: { source_type: 'guide', source_kind: 'style' },
        activation_policy: 'explicit_only',
        review_state: 'user_authored'
      },
      {
        path: 'reference/World Overview.md',
        kind: { source_type: 'reference', source_kind: 'world_summary' },
        activation_policy: 'explicit_only',
        review_state: 'user_authored'
      },
      {
        path: 'notes/brainstorm.md',
        kind: { source_type: 'note' },
        activation_policy: 'explicit_only',
        review_state: 'user_authored'
      }
    ]);
  });
});
