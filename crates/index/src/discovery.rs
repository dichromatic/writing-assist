use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use writing_assist_core::{DocumentType, ProjectDirectoryMapping, ProjectDirectoryRole};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredDocument {
    pub path: PathBuf,
    pub document_type: DocumentType,
}

fn document_type_for_role(role: &ProjectDirectoryRole) -> Option<DocumentType> {
    match role {
        ProjectDirectoryRole::PrimaryManuscript => Some(DocumentType::Manuscript),
        ProjectDirectoryRole::Reference => Some(DocumentType::Reference),
        ProjectDirectoryRole::Notes => Some(DocumentType::Note),
        ProjectDirectoryRole::Ignore => None,
    }
}

fn mapping_matches_path(path: &Path, root: &Path, mapping: &ProjectDirectoryMapping) -> bool {
    if !mapping.enabled {
        return false;
    }

    let Ok(relative_path) = path.strip_prefix(root) else {
        return false;
    };

    let mapping_path = Path::new(&mapping.path);
    // Classification is derived from configured directory roots, not from hardcoded folder names.
    relative_path.starts_with(mapping_path)
}

pub fn classify_document_path(
    path: &Path,
    root: &Path,
    mappings: &[ProjectDirectoryMapping],
) -> Option<DocumentType> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
        return None;
    }

    // Phase 1 import mappings are the source of truth for document typing.
    mappings
        .iter()
        .find(|mapping| mapping_matches_path(path, root, mapping))
        .and_then(|mapping| document_type_for_role(&mapping.role))
}

fn collect_markdown_files(directory: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if !directory.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recurse now so later span parsing can assume discovery already resolved nested drafts.
            collect_markdown_files(&path, files)?;
            continue;
        }

        if path.is_file() {
            files.push(path);
        }
    }

    Ok(())
}

pub fn discover_project_documents(
    root: &Path,
    mappings: &[ProjectDirectoryMapping],
) -> io::Result<Vec<DiscoveredDocument>> {
    let mut documents = Vec::new();
    let mut files = Vec::new();

    for mapping in mappings {
        if !mapping.enabled || mapping.role == ProjectDirectoryRole::Ignore {
            continue;
        }

        // Discovery only walks directories the user has explicitly mapped during project import.
        collect_markdown_files(&root.join(&mapping.path), &mut files)?;
    }

    // Keep discovery deterministic so UI ordering and tests do not depend on filesystem traversal order.
    files.sort();
    files.dedup();

    for path in files {
        let Some(document_type) = classify_document_path(&path, root, mappings) else {
            continue;
        };

        documents.push(DiscoveredDocument { path, document_type });
    }

    // Preserve stable ordering across import runs for the same project structure.
    documents.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(documents)
}
