mod chat;
mod context;
mod conversation;
mod documents;
mod evidence;
mod knowledge;
mod lexicon;
mod memory;
mod parsing;
mod preprocessing;
mod projects;
mod promoted_evidence;
mod tasks;

pub use chat::{ChatMessage, ChatMessageAuthor, ChatThread, ChatThreadScope};
pub use context::{
    ContextSource, ContextSourceActivationPolicy, ContextSourceKind, ContextSourceReviewState,
    GuideKind, ReferenceKind, classify_context_source_kind, context_source_allowed_by_default,
    context_source_included_by_default,
};
pub use conversation::ConversationMode;
pub use documents::{
    DocumentRecord, DocumentType, LoadedDocument, OpenedProject, ProjectDocumentEntry, SpanRecord,
};
pub use evidence::{
    DefinitionCandidate, EvidenceContext, MentionCandidate, MentionCluster, MentionClusterLink,
    MentionClusterLinkKind, MentionFeature, MentionOccurrence, SectionSummarySeed, SentenceType,
    StructuredFieldCandidate,
};
pub use knowledge::{
    DocumentArchetype, EntityProfileCandidate, RelationshipCandidate, StoryArcCandidate,
    StructuredKnowledgeCandidateKind, StructuredKnowledgeIntendedUse, TerminologyCandidate,
    TimelineEventCandidate, WorldRuleCandidate, structured_knowledge_intended_use,
};
pub use lexicon::{
    BootstrappedLexiconEntry, BootstrappedLexiconEntryKind, LexiconBootstrapRule,
    LexiconSupportRecord, LexiconSupportRecordKind,
};
pub use memory::{
    EntityCandidate, MemoryReviewState, MemorySourceReference, MemoryStalenessState,
    ReviewableFact, ReviewableSummary,
};
pub use parsing::{
    ParagraphParsingMode, ParsedMarkdownDocument, ParsedScene, ParsedSection, ParsedSpan,
    SectionBoundaryKind, SpanType,
};
pub use preprocessing::{
    PreprocessedDocument, PreprocessedQuoteSpan, PreprocessedSentence, PreprocessedSpan,
    PreprocessedToken, StructuralMarker, StructuralMarkerKind,
};
pub use projects::{
    ProjectConfig, ProjectConfigValidationError, ProjectDirectoryMapping, ProjectDirectoryRole,
    ProjectImportCandidate, ProjectImportSuggestionReason,
    normalize_project_directory_mapping_path, validate_project_directory_mappings,
};
pub use promoted_evidence::{
    EvidenceSuppressionReason, PromotedEvidenceBundle, PromotedEvidenceCandidate,
    PromotedEvidenceKind, PromotionReason, SuppressedEvidenceCandidate,
};
pub use tasks::{
    AnalysisComment, ContextBundle, DraftChange, IdeaCard, SelectionTarget,
    TASK_CONTRACT_SCHEMA_VERSION, TargetAnchor, TargetAnchorKind, TaskContractError, TaskOutput,
    TaskRequest, TaskResult, TaskType,
};
