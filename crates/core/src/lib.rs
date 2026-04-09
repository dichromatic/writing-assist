use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;
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
pub struct ProjectConfig {
    // Persist the normalized project root so reopening the same project can reuse saved mappings.
    pub root_path: String,
    pub directory_mappings: Vec<ProjectDirectoryMapping>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectDocumentEntry {
    // Keep this relative to the project root so the UI can use it directly for file tree rendering later.
    pub path: String,
    pub document_type: DocumentType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenedProject {
    pub config: ProjectConfig,
    pub documents: Vec<ProjectDocumentEntry>,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ProjectConfigValidationError {
    #[error("directory mapping paths must not be empty")]
    EmptyDirectoryPath,
    #[error("duplicate directory mapping path: {0}")]
    DuplicateDirectoryPath(String),
    #[error("exactly one enabled primary manuscript directory is required")]
    InvalidPrimaryManuscriptCount,
}

pub fn validate_project_directory_mappings(
    directory_mappings: &[ProjectDirectoryMapping],
) -> Result<(), ProjectConfigValidationError> {
    let mut seen_paths = HashSet::new();
    let mut primary_manuscript_count = 0;

    for mapping in directory_mappings {
        let normalized_path = mapping.path.trim();

        if normalized_path.is_empty() {
            return Err(ProjectConfigValidationError::EmptyDirectoryPath);
        }

        // Persistence rejects duplicates up front so later discovery does not depend on DB conflict handling.
        if !seen_paths.insert(normalized_path.to_string()) {
            return Err(ProjectConfigValidationError::DuplicateDirectoryPath(
                normalized_path.to_string(),
            ));
        }

        if mapping.enabled && mapping.role == ProjectDirectoryRole::PrimaryManuscript {
            primary_manuscript_count += 1;
        }
    }

    if primary_manuscript_count != 1 {
        return Err(ProjectConfigValidationError::InvalidPrimaryManuscriptCount);
    }

    Ok(())
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
    use super::{
        validate_project_directory_mappings, ProjectConfigValidationError, ProjectDirectoryMapping,
        ProjectDirectoryRole, ProjectImportSuggestionReason,
    };

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

    #[test]
    fn rejects_duplicate_mapping_paths() {
        let result = validate_project_directory_mappings(&[
            ProjectDirectoryMapping {
                path: "drafts".to_string(),
                role: ProjectDirectoryRole::PrimaryManuscript,
                enabled: true,
            },
            ProjectDirectoryMapping {
                path: "drafts".to_string(),
                role: ProjectDirectoryRole::Reference,
                enabled: true,
            },
        ]);

        assert_eq!(
            result,
            Err(ProjectConfigValidationError::DuplicateDirectoryPath(
                "drafts".to_string()
            ))
        );
    }

    #[test]
    fn rejects_missing_primary_manuscript_directory() {
        let result = validate_project_directory_mappings(&[ProjectDirectoryMapping {
            path: "notes".to_string(),
            role: ProjectDirectoryRole::Notes,
            enabled: true,
        }]);

        assert_eq!(
            result,
            Err(ProjectConfigValidationError::InvalidPrimaryManuscriptCount)
        );
    }

    #[test]
    fn accepts_one_enabled_primary_manuscript_directory() {
        let result = validate_project_directory_mappings(&[
            ProjectDirectoryMapping {
                path: "drafts".to_string(),
                role: ProjectDirectoryRole::PrimaryManuscript,
                enabled: true,
            },
            ProjectDirectoryMapping {
                path: "notes".to_string(),
                role: ProjectDirectoryRole::Notes,
                enabled: false,
            },
        ]);

        assert_eq!(result, Ok(()));
    }
}
