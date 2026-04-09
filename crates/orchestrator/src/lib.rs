use std::path::Path;

use thiserror::Error;
use writing_assist_core::{OpenedProject, ProjectDocumentEntry};

pub fn phase_zero_status() -> &'static str {
    "scaffolded"
}

#[derive(Debug, Error)]
pub enum OpenProjectError {
    #[error(transparent)]
    Store(#[from] writing_assist_store::StoreError),
    #[error(transparent)]
    Discovery(#[from] std::io::Error),
    #[error("project has no saved import configuration")]
    NotConfigured,
    #[error("discovered document path was not relative to the saved project root: {0}")]
    NonRelativeDiscoveredPath(String),
}

pub async fn open_configured_project(project_root: &Path) -> Result<OpenedProject, OpenProjectError> {
    let Some(config) = writing_assist_store::load_project_config(project_root).await? else {
        return Err(OpenProjectError::NotConfigured);
    };

    let config_root = Path::new(&config.root_path);
    let discovered_documents =
        writing_assist_index::discover_project_documents(config_root, &config.directory_mappings)?;

    let mut documents = Vec::with_capacity(discovered_documents.len());

    for document in discovered_documents {
        let relative_path = document
            .path
            .strip_prefix(config_root)
            .map_err(|_| {
                OpenProjectError::NonRelativeDiscoveredPath(
                    document.path.to_string_lossy().to_string(),
                )
            })?
            .to_string_lossy()
            .to_string();

        // Phase 1.5 is the first point where persisted import config actively drives project reopen behavior.
        documents.push(ProjectDocumentEntry {
            path: relative_path,
            document_type: document.document_type,
        });
    }

    Ok(OpenedProject { config, documents })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use tempfile::tempdir;
    use writing_assist_core::{DocumentType, ProjectDirectoryMapping, ProjectDirectoryRole};

    use super::{open_configured_project, OpenProjectError};

    fn mapping(path: &str, role: ProjectDirectoryRole) -> ProjectDirectoryMapping {
        ProjectDirectoryMapping {
            path: path.to_string(),
            role,
            enabled: true,
        }
    }

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent directories should be created");
        }

        fs::write(path, contents).expect("test file should be written");
    }

    #[tokio::test]
    async fn opens_project_from_saved_configuration_and_discovers_documents() {
        let project_root = tempdir().expect("project root");

        write_file(&project_root.path().join("drafts/chapter 1.md"), "# Chapter 1");
        write_file(&project_root.path().join("drafts/chapter 2.md"), "# Chapter 2");
        write_file(&project_root.path().join("lore/history.md"), "# History");
        write_file(&project_root.path().join("notes/scratch.txt"), "ignored");

        writing_assist_store::save_project_config(
            project_root.path(),
            &[
                mapping("drafts", ProjectDirectoryRole::PrimaryManuscript),
                mapping("lore", ProjectDirectoryRole::Reference),
            ],
        )
        .await
        .expect("save project config");

        let opened = open_configured_project(project_root.path())
            .await
            .expect("open configured project");

        assert_eq!(opened.documents.len(), 3);
        assert_eq!(opened.documents[0].path, "drafts/chapter 1.md");
        assert_eq!(opened.documents[0].document_type, DocumentType::Manuscript);
        assert_eq!(opened.documents[1].path, "drafts/chapter 2.md");
        assert_eq!(opened.documents[1].document_type, DocumentType::Manuscript);
        assert_eq!(opened.documents[2].path, "lore/history.md");
        assert_eq!(opened.documents[2].document_type, DocumentType::Reference);
    }

    #[tokio::test]
    async fn returns_not_configured_for_projects_without_saved_import_config() {
        let project_root = tempdir().expect("project root");

        let error = open_configured_project(project_root.path())
            .await
            .expect_err("project should not open without saved config");

        assert!(matches!(error, OpenProjectError::NotConfigured));
    }
}
