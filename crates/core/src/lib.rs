use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConversationMode {
    Analysis,
    Editing,
    Ideation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentType {
    Manuscript,
    Reference,
    Note,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectDirectoryRole {
    // The user's main writing directory. Phase 1 import validation requires exactly one.
    PrimaryManuscript,
    Reference,
    Notes,
    Ignore,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectDirectoryMapping {
    // Store a project-relative path so the mapping remains stable if the project root moves.
    pub path: String,
    pub role: ProjectDirectoryRole,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectImportSuggestionReason {
    ContainsMarkdownFiles,
    DirectoryNamedChapters,
    DirectoryNamedWorldContext,
    DirectoryNamedNotes,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectImportCandidate {
    // Keep this relative to the selected project root so the import UI can persist the mapping directly.
    pub path: String,
    pub contains_markdown_files: bool,
    pub suggested_role: Option<ProjectDirectoryRole>,
    pub suggestion_reasons: Vec<ProjectImportSuggestionReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpanType {
    Heading,
    Paragraph,
    Section,
    Window,
    Scene,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: Uuid,
    pub path: String,
    pub document_type: DocumentType,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanRecord {
    pub id: Uuid,
    pub document_id: Uuid,
    pub span_type: SpanType,
    pub ordinal: i32,
}
