use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConversationMode {
    Analysis,
    Editing,
    Ideation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentType {
    Manuscript,
    Reference,
    Note,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
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
#[serde(rename_all = "snake_case")]
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
#[serde(rename_all = "snake_case")]
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

#[cfg(test)]
mod tests {
    use super::{ProjectDirectoryRole, ProjectImportSuggestionReason};

    #[test]
    fn serializes_directory_roles_as_snake_case() {
        let serialized =
            serde_json::to_string(&ProjectDirectoryRole::PrimaryManuscript).expect("serialize");

        // Frontend import state uses snake_case role identifiers, so the backend contract must match.
        assert_eq!(serialized, "\"primary_manuscript\"");
    }

    #[test]
    fn serializes_import_suggestion_reasons_as_snake_case() {
        let serialized = serde_json::to_string(
            &ProjectImportSuggestionReason::DirectoryNamedWorldContext,
        )
        .expect("serialize");

        assert_eq!(serialized, "\"directory_named_world_context\"");
    }
}
