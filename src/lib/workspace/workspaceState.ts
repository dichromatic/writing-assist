import type { TaskTargetAnchor } from '$lib/task/targets';

export type PaneId = string;

export type PaneSelection = {
  paneId: PaneId;
  documentPath: string;
  selectedText: string;
  startChar: number;
  endChar: number;
  overlappingSpanOrdinals: number[];
  primarySpanOrdinal: number | null;
  documentContentHash?: string;
};

export type ActiveTaskTarget = {
  sourcePaneId: PaneId;
  documentPath: string;
  selectedText: string;
  startChar: number;
  endChar: number;
  anchors: TaskTargetAnchor[];
  documentContentHash?: string;
};

export type WorkspacePane =
  | { paneType: 'primary_editor'; paneId: PaneId; documentPath: string | null }
  | { paneType: 'reference_editor'; paneId: PaneId; documentPath: string }
  | { paneType: 'intelligence_hub'; paneId: PaneId }
  | { paneType: 'knowledge_rail'; paneId: PaneId };

export type WorkspaceState = {
  focusedPaneId: PaneId | null;
  activeSelectionPaneId: PaneId | null;
  panes: WorkspacePane[];
  selectionByPaneId: Record<PaneId, PaneSelection>;
};

export function createInitialWorkspaceState(primaryDocumentPath: string | null = null): WorkspaceState {
  return {
    focusedPaneId: 'primary',
    activeSelectionPaneId: null,
    panes: [
      { paneType: 'primary_editor', paneId: 'primary', documentPath: primaryDocumentPath },
      { paneType: 'intelligence_hub', paneId: 'intelligence' },
      { paneType: 'knowledge_rail', paneId: 'knowledge' }
    ],
    selectionByPaneId: {}
  };
}

export function setFocusedPane(workspace: WorkspaceState, paneId: PaneId | null): WorkspaceState {
  return {
    ...workspace,
    focusedPaneId: paneId
  };
}

export function updatePaneSelection(
  workspace: WorkspaceState,
  selection: PaneSelection
): WorkspaceState {
  return {
    ...workspace,
    // The registry keeps each pane's latest selection so the intelligence hub never has to query editor internals.
    selectionByPaneId: {
      ...workspace.selectionByPaneId,
      [selection.paneId]: selection
    },
    // Keep the task target tied to the last editor selection even after the user moves focus into chat.
    activeSelectionPaneId: selection.paneId,
    focusedPaneId: selection.paneId
  };
}

export function getActiveTaskTarget(workspace: WorkspaceState): ActiveTaskTarget | null {
  if (!workspace.activeSelectionPaneId) {
    return null;
  }

  const selection = workspace.selectionByPaneId[workspace.activeSelectionPaneId];

  if (!selection?.selectedText) {
    return null;
  }

  const anchors: TaskTargetAnchor[] = selection.overlappingSpanOrdinals.map((ordinal) => ({
    kind: 'span',
    ordinal
  }));

  return {
    sourcePaneId: selection.paneId,
    documentPath: selection.documentPath,
    selectedText: selection.selectedText,
    startChar: selection.startChar,
    endChar: selection.endChar,
    anchors,
    documentContentHash: selection.documentContentHash
  };
}

export function pinReferencePane(workspace: WorkspaceState, documentPath: string): WorkspaceState {
  const paneId = `reference:${documentPath}`;

  if (workspace.panes.some((pane) => pane.paneId === paneId)) {
    return workspace;
  }

  return {
    ...workspace,
    panes: [
      ...workspace.panes,
      {
        paneType: 'reference_editor',
        paneId,
        documentPath
      }
    ]
  };
}
