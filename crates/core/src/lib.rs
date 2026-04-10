mod context;
mod conversation;
mod documents;
mod parsing;
mod projects;

pub use context::{
    context_source_allowed_by_default, ContextSource, ContextSourceActivationPolicy,
    ContextSourceKind, ContextSourceReviewState, GuideKind, ReferenceKind,
};
pub use conversation::ConversationMode;
pub use documents::{
    DocumentRecord, DocumentType, LoadedDocument, OpenedProject, ProjectDocumentEntry, SpanRecord,
};
pub use parsing::{
    ParagraphParsingMode, ParsedMarkdownDocument, ParsedScene, ParsedSection, ParsedSpan,
    SectionBoundaryKind, SpanType,
};
pub use projects::{
    validate_project_directory_mappings, ProjectConfig, ProjectConfigValidationError,
    ProjectDirectoryMapping, ProjectDirectoryRole, ProjectImportCandidate,
    ProjectImportSuggestionReason,
};
