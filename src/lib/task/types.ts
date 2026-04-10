import type { TaskSelectionTarget } from '$lib/project-import/selection';

export type ConversationMode = 'analysis' | 'editing' | 'ideation';

export type TaskType = 'analyze_selection' | 'rewrite_selection' | 'ideate_selection' | 'chat';

export type GuideKind = 'prose' | 'style' | 'critique' | 'rewrite' | 'custom';

export type ReferenceKind =
  | 'story_summary'
  | 'world_summary'
  | 'character_bible'
  | 'timeline'
  | 'terminology'
  | 'research'
  | 'custom';

export type ContextSourceKind =
  | { source_type: 'guide'; source_kind: GuideKind }
  | { source_type: 'reference'; source_kind: ReferenceKind }
  | { source_type: 'note' };

export type ContextSource = {
  path: string;
  kind: ContextSourceKind;
  activation_policy: 'pinned' | 'retrieved' | 'explicit_only';
  review_state: 'user_authored' | 'pending_review' | 'approved' | 'stale';
};

export type DeterministicTaskCommandRequest = {
  mode: ConversationMode;
  task_type: TaskType;
  target: TaskSelectionTarget;
  available_sources: ContextSource[];
  explicitly_selected_source_paths: string[];
};

export type AnalysisComment = {
  target: TaskSelectionTarget;
  message: string;
};

export type DraftChange = {
  target: TaskSelectionTarget;
  original_text: string;
  proposed_text: string;
};

export type IdeaCard = {
  title: string;
  body: string;
};

export type TaskOutput =
  | { output_type: 'analysis_comment'; output: AnalysisComment }
  | { output_type: 'draft_change'; output: DraftChange }
  | { output_type: 'idea_card'; output: IdeaCard };

export type TaskResult = {
  request_id: string;
  schema_version: number;
  mode: ConversationMode;
  outputs: TaskOutput[];
};
