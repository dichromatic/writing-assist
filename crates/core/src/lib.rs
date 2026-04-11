mod chat;
mod context;
mod conversation;
mod documents;
mod evidence;
mod knowledge;
mod memory;
mod parsing;
mod tasks;
mod projects;

pub use chat::{ChatMessage, ChatMessageAuthor, ChatThread, ChatThreadScope};
pub use context::{
    classify_context_source_kind, context_source_allowed_by_default,
    context_source_included_by_default, ContextSource, ContextSourceActivationPolicy,
    ContextSourceKind, ContextSourceReviewState, GuideKind, ReferenceKind,
};
pub use conversation::ConversationMode;
pub use documents::{
    DocumentRecord, DocumentType, LoadedDocument, OpenedProject, ProjectDocumentEntry, SpanRecord,
};
pub use evidence::{
    DefinitionCandidate, EvidenceContext, MentionCandidate, MentionFeature,
    SectionSummarySeed, StructuredFieldCandidate,
};
pub use knowledge::{
    structured_knowledge_intended_use, DocumentArchetype, EntityProfileCandidate,
    RelationshipCandidate, StoryArcCandidate, StructuredKnowledgeCandidateKind,
    StructuredKnowledgeIntendedUse, TerminologyCandidate, TimelineEventCandidate,
    WorldRuleCandidate,
};
pub use memory::{
    EntityCandidate, MemoryReviewState, MemorySourceReference, MemoryStalenessState,
    ReviewableFact, ReviewableSummary,
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
