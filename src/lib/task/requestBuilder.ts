import {
  toTaskSelectionTarget,
  type DocumentSelectionTarget,
  type SelectionTargetAdapterError
} from '$lib/project-import/selection';
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

export function buildDeterministicTaskRequest(
  mode: ConversationMode,
  selection: DocumentSelectionTarget
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
      available_sources: [],
      explicitly_selected_source_paths: []
    }
  };
}
