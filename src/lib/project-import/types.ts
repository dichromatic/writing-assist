export const projectDirectoryRoles = [
  'primary_manuscript',
  'reference',
  'notes',
  'ignore'
] as const;

// Keep the frontend identifiers aligned with the snake_case serde contract from crates/core.
export type ProjectDirectoryRole = (typeof projectDirectoryRoles)[number];

export type ProjectImportSuggestionReason =
  | 'contains_markdown_files'
  | 'directory_named_chapters'
  | 'directory_named_world_context'
  | 'directory_named_notes';

export type ProjectImportCandidate = {
  path: string;
  contains_markdown_files: boolean;
  suggested_role: ProjectDirectoryRole | null;
  suggestion_reasons: ProjectImportSuggestionReason[];
};

// This draft shape mirrors the persisted mapping model Phase 1.4 will store from the import UI.
export type ProjectDirectoryMappingDraft = {
  path: string;
  role: ProjectDirectoryRole | null;
  enabled: boolean;
};
