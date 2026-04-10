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

export type PersistedProjectDirectoryMapping = {
  path: string;
  role: ProjectDirectoryRole;
  enabled: boolean;
};

export type ProjectConfig = {
  root_path: string;
  directory_mappings: PersistedProjectDirectoryMapping[];
};

export type ProjectDocumentEntry = {
  path: string;
  document_type: 'manuscript' | 'reference' | 'note';
};

export type OpenedProject = {
  config: ProjectConfig;
  documents: ProjectDocumentEntry[];
};

export type SpanType = 'heading' | 'paragraph' | 'section' | 'window' | 'scene';

export type SectionBoundaryKind = 'file_start' | 'heading' | 'scene_break';

export type ParsedSpan = {
  ordinal: number;
  span_type: SpanType;
  text: string;
  normalized_text: string;
  start_line: number;
  end_line: number;
  start_byte: number;
  end_byte: number;
  start_char: number;
  end_char: number;
};

export type ParsedSection = {
  ordinal: number;
  text: string;
  normalized_text: string;
  boundary_kind: SectionBoundaryKind;
  boundary_text: string | null;
  start_line: number;
  end_line: number;
  start_byte: number;
  end_byte: number;
  start_char: number;
  end_char: number;
};

export type ParsedScene = {
  ordinal: number;
  text: string;
  normalized_text: string;
  separator: string | null;
  start_line: number;
  end_line: number;
  start_byte: number;
  end_byte: number;
  start_char: number;
  end_char: number;
  start_span_ordinal: number;
  end_span_ordinal: number;
};

export type ParsedMarkdownDocument = {
  spans: ParsedSpan[];
  sections: ParsedSection[];
  scenes: ParsedScene[];
};

export type LoadedDocument = {
  document: ProjectDocumentEntry;
  markdown: string;
  parsed: ParsedMarkdownDocument;
};
