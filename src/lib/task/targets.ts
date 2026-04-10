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
