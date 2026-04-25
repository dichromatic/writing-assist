import { describe, expect, it } from 'vitest';

import {
  createKnowledgeRailState,
  setKnowledgeRailActivePaths,
  toggleKnowledgeRailPath
} from './knowledgeRail';
import type { ContextSource } from '$lib/task/types';

function source(path: string): ContextSource {
  return {
    path,
    kind: { source_type: 'guide', source_kind: 'prose' },
    activation_policy: 'pinned',
    review_state: 'approved'
  };
}

describe('knowledge rail state', () => {
  it('starts with no active context paths', () => {
    expect(createKnowledgeRailState([source('guides/prose.md')])).toEqual({
      availableSources: [source('guides/prose.md')],
      activeContextPaths: []
    });
  });

  it('toggles only known source paths', () => {
    const state = toggleKnowledgeRailPath(
      toggleKnowledgeRailPath(createKnowledgeRailState([source('guides/prose.md')]), 'guides/prose.md'),
      'unknown.md'
    );

    expect(state.activeContextPaths).toEqual(['guides/prose.md']);
  });

  it('deduplicates and filters active paths when setting them directly', () => {
    const state = setKnowledgeRailActivePaths(
      createKnowledgeRailState([source('guides/prose.md'), source('reference/world-summary.md')]),
      ['guides/prose.md', 'guides/prose.md', 'missing.md', 'reference/world-summary.md']
    );

    expect(state.activeContextPaths).toEqual([
      'guides/prose.md',
      'reference/world-summary.md'
    ]);
  });
});
