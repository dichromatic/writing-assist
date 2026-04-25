import { describe, expect, it } from 'vitest';

import {
  createInitialWorkspaceState,
  getActiveTaskTarget,
  pinReferencePane,
  setFocusedPane,
  updatePaneSelection,
  type PaneSelection
} from './workspaceState';

function selection(overrides: Partial<PaneSelection> = {}): PaneSelection {
  return {
    paneId: 'primary',
    documentPath: 'chapters/chapter-1.md',
    selectedText: 'The selected sentence.',
    startChar: 12,
    endChar: 34,
    overlappingSpanOrdinals: [3, 4],
    primarySpanOrdinal: 3,
    documentContentHash: 'hash-1',
    ...overrides
  };
}

describe('workspace state', () => {
  it('starts with primary editor, intelligence hub, and knowledge rail panes', () => {
    const workspace = createInitialWorkspaceState('chapters/chapter-1.md');

    expect(workspace.focusedPaneId).toBe('primary');
    expect(workspace.panes).toEqual([
      { paneType: 'primary_editor', paneId: 'primary', documentPath: 'chapters/chapter-1.md' },
      { paneType: 'intelligence_hub', paneId: 'intelligence' },
      { paneType: 'knowledge_rail', paneId: 'knowledge' }
    ]);
  });

  it('derives the active task target from the focused pane selection', () => {
    const workspace = updatePaneSelection(
      createInitialWorkspaceState('chapters/chapter-1.md'),
      selection()
    );

    expect(getActiveTaskTarget(workspace)).toEqual({
      sourcePaneId: 'primary',
      documentPath: 'chapters/chapter-1.md',
      selectedText: 'The selected sentence.',
      startChar: 12,
      endChar: 34,
      anchors: [
        { kind: 'span', ordinal: 3 },
        { kind: 'span', ordinal: 4 }
      ],
      documentContentHash: 'hash-1'
    });
  });

  it('keeps the active task target when focus moves from the editor into the intelligence hub', () => {
    const workspace = setFocusedPane(
      updatePaneSelection(createInitialWorkspaceState('chapters/chapter-1.md'), selection()),
      'intelligence'
    );

    expect(getActiveTaskTarget(workspace)).toEqual({
      sourcePaneId: 'primary',
      documentPath: 'chapters/chapter-1.md',
      selectedText: 'The selected sentence.',
      startChar: 12,
      endChar: 34,
      anchors: [
        { kind: 'span', ordinal: 3 },
        { kind: 'span', ordinal: 4 }
      ],
      documentContentHash: 'hash-1'
    });
  });

  it('does not guess a task target when no editor selection has ever been activated', () => {
    const workspace = setFocusedPane(createInitialWorkspaceState('chapters/chapter-1.md'), 'intelligence');

    expect(getActiveTaskTarget(workspace)).toBeNull();
  });

  it('can pin a reference document without duplicating an existing reference pane', () => {
    const workspace = pinReferencePane(
      pinReferencePane(createInitialWorkspaceState('chapters/chapter-1.md'), 'world/summary.md'),
      'world/summary.md'
    );

    expect(workspace.panes.filter((pane) => pane.paneType === 'reference_editor')).toEqual([
      {
        paneType: 'reference_editor',
        paneId: 'reference:world/summary.md',
        documentPath: 'world/summary.md'
      }
    ]);
  });
});
