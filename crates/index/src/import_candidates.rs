use std::fs;
use std::io;
use std::path::Path;

use writing_assist_core::{
    ProjectDirectoryRole, ProjectImportCandidate, ProjectImportSuggestionReason,
};

use crate::project_files::{is_hidden_or_app_directory, is_supported_markdown_file};

fn has_direct_markdown_files(directory: &Path) -> io::Result<bool> {
    if !directory.exists() {
        return Ok(false);
    }

    for entry in fs::read_dir(directory)? {
        let path = entry?.path();

        if path.is_file() && is_supported_markdown_file(&path) {
            return Ok(true);
        }
    }

    Ok(false)
}

fn has_markdown_files(directory: &Path) -> io::Result<bool> {
    if !directory.exists() {
        return Ok(false);
    }

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            if is_hidden_or_app_directory(&path) {
                continue;
            }

            if has_markdown_files(&path)? {
                return Ok(true);
            }
        }

        if path.is_file() && is_supported_markdown_file(&path) {
            return Ok(true);
        }
    }

    Ok(false)
}

fn suggested_role_for_directory_name(
    directory_name: &str,
) -> (Option<ProjectDirectoryRole>, Vec<ProjectImportSuggestionReason>) {
    match directory_name {
        "chapters" => (
            Some(ProjectDirectoryRole::PrimaryManuscript),
            vec![ProjectImportSuggestionReason::DirectoryNamedChapters],
        ),
        "world_context" => (
            Some(ProjectDirectoryRole::Reference),
            vec![ProjectImportSuggestionReason::DirectoryNamedWorldContext],
        ),
        "notes" => (
            Some(ProjectDirectoryRole::Notes),
            vec![ProjectImportSuggestionReason::DirectoryNamedNotes],
        ),
        _ => (None, Vec::new()),
    }
}

pub fn discover_project_import_candidates(root: &Path) -> io::Result<Vec<ProjectImportCandidate>> {
    let mut candidates = Vec::new();

    if has_direct_markdown_files(root)? {
        candidates.push(ProjectImportCandidate {
            path: ".".to_string(),
            contains_markdown_files: true,
            suggested_role: None,
            suggestion_reasons: vec![ProjectImportSuggestionReason::ContainsMarkdownFiles],
        });
    }

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() || is_hidden_or_app_directory(&path) {
            continue;
        }

        let Some(directory_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        let contains_markdown_files = has_markdown_files(&path)?;
        let (suggested_role, mut suggestion_reasons) =
            suggested_role_for_directory_name(directory_name);

        if contains_markdown_files {
            // Markdown presence is useful import context even when the directory name is not meaningful.
            suggestion_reasons.push(ProjectImportSuggestionReason::ContainsMarkdownFiles);
        }

        candidates.push(ProjectImportCandidate {
            path: directory_name.to_string(),
            contains_markdown_files,
            suggested_role,
            suggestion_reasons,
        });
    }

    // Stable ordering keeps the import UI deterministic for the same project root.
    candidates.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(candidates)
}
