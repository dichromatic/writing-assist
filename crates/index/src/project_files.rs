use std::path::Path;

use writing_assist_core::ProjectDirectoryRole;

pub(crate) fn is_supported_project_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "md" | "markdown" | "mdown" | "txt"
            )
        })
        .unwrap_or(false)
}

pub(crate) fn is_supported_file_for_role(path: &Path, role: &ProjectDirectoryRole) -> bool {
    match role {
        ProjectDirectoryRole::PrimaryManuscript => path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|extension| {
                matches!(extension.to_ascii_lowercase().as_str(), "md" | "markdown" | "mdown")
            })
            .unwrap_or(false),
        ProjectDirectoryRole::Reference | ProjectDirectoryRole::Notes => {
            is_supported_project_text_file(path)
        }
        ProjectDirectoryRole::Ignore => false,
    }
}

pub(crate) fn is_hidden_or_app_directory(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with('.'))
        .unwrap_or(false)
}
