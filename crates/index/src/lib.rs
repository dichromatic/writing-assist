use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use writing_assist_core::{DocumentType, SpanType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredDocument {
    pub path: PathBuf,
    pub document_type: DocumentType,
}

pub fn supported_span_types() -> [SpanType; 5] {
    [
        SpanType::Heading,
        SpanType::Paragraph,
        SpanType::Section,
        SpanType::Window,
        SpanType::Scene,
    ]
}

pub fn classify_document_path(path: &Path, root: &Path) -> Option<DocumentType> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
        return None;
    }

    let relative_path = path.strip_prefix(root).ok()?;
    let top_level = relative_path.iter().next()?.to_str()?;

    match top_level {
        "chapters" => Some(DocumentType::Chapter),
        "world_context" => Some(DocumentType::Reference),
        _ => None,
    }
}

pub fn discover_project_documents(root: &Path) -> io::Result<Vec<DiscoveredDocument>> {
    let mut documents = Vec::new();

    for folder in ["chapters", "world_context"] {
        let directory = root.join(folder);

        if !directory.exists() {
            continue;
        }

        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let Some(document_type) = classify_document_path(&path, root) else {
                continue;
            };

            documents.push(DiscoveredDocument { path, document_type });
        }
    }

    documents.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(documents)
}
