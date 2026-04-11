import {
  toTaskSelectionTarget,
  type DocumentSelectionTarget,
  type SelectionTargetAdapterError
} from '$lib/project-import/selection';
import type { ContextSource } from '$lib/task/types';
import type {
  ConversationMode,
  DeterministicTaskCommandRequest,
  TaskType
} from '$lib/task/types';

const taskTypeByMode: Record<ConversationMode, TaskType> = {
  analysis: 'analyze_selection',
  editing: 'rewrite_selection',
  ideation: 'ideate_selection'
};

export type DeterministicTaskRequestBuildResult =
  | { ok: true; request: DeterministicTaskCommandRequest }
  | { ok: false; error: SelectionTargetAdapterError; message: string };

export type TaskContextSelection = {
  availableSources?: ContextSource[];
  activeContextPaths?: string[];
};

export function buildDeterministicTaskRequest(
  mode: ConversationMode,
  selection: DocumentSelectionTarget,
  contextSelection: TaskContextSelection = {}
): DeterministicTaskRequestBuildResult {
  // Keep all frontend task execution behind the Phase 2.2 selection adapter.
  const targetResult = toTaskSelectionTarget(selection);

  if (!targetResult.ok) {
    return targetResult;
  }

  return {
    ok: true,
    request: {
      mode,
      task_type: taskTypeByMode[mode],
      target: targetResult.target,
      available_sources: contextSelection.availableSources ?? [],
      explicitly_selected_source_paths: contextSelection.activeContextPaths ?? []
    }
  };
}
