use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::parsing::ParsedMarkdownDocument;
use crate::projects::ProjectConfig;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentType {
    Manuscript,
    Reference,
    Note,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadedDocument {
    pub document: ProjectDocumentEntry,
    // Keep raw Markdown in the load payload so CodeMirror can render the exact source file content.
    pub markdown: String,
    // Parsed spans are sent alongside the raw text so later selection/comment UI can anchor to parser ranges.
    pub parsed: ParsedMarkdownDocument,
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
    pub span_type: crate::parsing::SpanType,
    pub ordinal: i32,
}
