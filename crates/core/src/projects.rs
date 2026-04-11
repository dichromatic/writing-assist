use std::collections::HashSet;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

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

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ProjectConfigValidationError {
    #[error("directory mapping paths must not be empty")]
    EmptyDirectoryPath,
    #[error("directory mapping path must stay inside the project root: {0}")]
    UnsafeDirectoryPath(String),
    #[error("duplicate directory mapping path: {0}")]
    DuplicateDirectoryPath(String),
    #[error("exactly one enabled primary manuscript directory is required")]
    InvalidPrimaryManuscriptCount,
}

pub fn normalize_project_directory_mapping_path(
    path: &str,
) -> Result<String, ProjectConfigValidationError> {
    let trimmed_path = path.trim();

    if trimmed_path.is_empty() {
        return Err(ProjectConfigValidationError::EmptyDirectoryPath);
    }

    let parsed_path = Path::new(trimmed_path);

    if parsed_path.is_absolute() {
        return Err(ProjectConfigValidationError::UnsafeDirectoryPath(
            trimmed_path.to_string(),
        ));
    }

    let mut normalized_components = Vec::new();

    for component in parsed_path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(component) => {
                normalized_components.push(component.to_string_lossy().to_string());
            }
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ProjectConfigValidationError::UnsafeDirectoryPath(
                    trimmed_path.to_string(),
                ));
            }
        }
    }

    if normalized_components.is_empty() {
        return Ok(".".to_string());
    }

    // Mapping paths are persisted with slash separators so UI/backend comparisons are stable.
    Ok(normalized_components.join("/"))
}

pub fn validate_project_directory_mappings(
    directory_mappings: &[ProjectDirectoryMapping],
) -> Result<(), ProjectConfigValidationError> {
    let mut seen_paths = HashSet::new();
    let mut primary_manuscript_count = 0;

    for mapping in directory_mappings {
        let normalized_path = normalize_project_directory_mapping_path(&mapping.path)?;

        // Persistence rejects duplicates up front so later discovery does not depend on DB conflict handling.
        if !seen_paths.insert(normalized_path.clone()) {
            return Err(ProjectConfigValidationError::DuplicateDirectoryPath(
                normalized_path,
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
    ContainsSupportedTextFiles,
    DirectoryNamedChapters,
    DirectoryNamedWorldContext,
    DirectoryNamedNotes,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectImportCandidate {
    // Keep this relative to the selected project root so the import UI can persist the mapping directly.
    pub path: String,
    pub contains_supported_text_files: bool,
    pub suggested_role: Option<ProjectDirectoryRole>,
    pub suggestion_reasons: Vec<ProjectImportSuggestionReason>,
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
    fn accepts_root_directory_mapping_for_root_level_manuscript_files() {
        let result = validate_project_directory_mappings(&[ProjectDirectoryMapping {
            path: ".".to_string(),
            role: ProjectDirectoryRole::PrimaryManuscript,
            enabled: true,
        }]);

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn rejects_mapping_paths_that_escape_the_project_root() {
        let result = validate_project_directory_mappings(&[ProjectDirectoryMapping {
            path: "../outside".to_string(),
            role: ProjectDirectoryRole::PrimaryManuscript,
            enabled: true,
        }]);

        assert_eq!(
            result,
            Err(ProjectConfigValidationError::UnsafeDirectoryPath(
                "../outside".to_string()
            ))
        );
    }

    #[test]
    fn rejects_absolute_mapping_paths() {
        let result = validate_project_directory_mappings(&[ProjectDirectoryMapping {
            path: "/tmp/outside".to_string(),
            role: ProjectDirectoryRole::PrimaryManuscript,
            enabled: true,
        }]);

        assert_eq!(
            result,
            Err(ProjectConfigValidationError::UnsafeDirectoryPath(
                "/tmp/outside".to_string()
            ))
        );
    }

    #[test]
    fn detects_duplicate_paths_after_normalization() {
        let result = validate_project_directory_mappings(&[
            ProjectDirectoryMapping {
                path: "drafts".to_string(),
                role: ProjectDirectoryRole::PrimaryManuscript,
                enabled: true,
            },
            ProjectDirectoryMapping {
                path: "drafts/".to_string(),
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
