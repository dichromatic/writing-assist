mod chat;
mod context;
mod conversation;
mod documents;
mod parsing;
mod tasks;
mod projects;

pub use chat::{ChatMessage, ChatMessageAuthor, ChatThread, ChatThreadScope};
pub use context::{
    context_source_allowed_by_default, context_source_included_by_default, ContextSource,
    ContextSourceActivationPolicy, ContextSourceKind, ContextSourceReviewState, GuideKind,
    ReferenceKind,
};
pub use conversation::ConversationMode;
pub use documents::{
    DocumentRecord, DocumentType, LoadedDocument, OpenedProject, ProjectDocumentEntry, SpanRecord,
};
pub use tasks::{
    AnalysisComment, ContextBundle, DraftChange, IdeaCard, TaskContractError, TaskOutput,
    TaskRequest, TaskResult, TaskType, SelectionTarget, TargetAnchor, TargetAnchorKind,
    TASK_CONTRACT_SCHEMA_VERSION,
};
pub use parsing::{
    ParagraphParsingMode, ParsedMarkdownDocument, ParsedScene, ParsedSection, ParsedSpan,
    SectionBoundaryKind, SpanType,
};
pub use projects::{
    normalize_project_directory_mapping_path, validate_project_directory_mappings, ProjectConfig,
    ProjectConfigValidationError, ProjectDirectoryMapping, ProjectDirectoryRole,
    ProjectImportCandidate, ProjectImportSuggestionReason,
};
