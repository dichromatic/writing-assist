use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanType {
    Heading,
    Paragraph,
    // Sections are targetable parse objects, but the current parser keeps them in `ParsedSection`.
    Section,
    // Windows are reserved for Phase 2 context assembly and are not emitted by the Phase 1 parser.
    Window,
    Scene,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParagraphParsingMode {
    // Only blank lines end paragraphs; no implicit paragraph splitting is attempted.
    StrictBlankLines,
    // Blank lines remain the primary signal, but a narrow fallback heuristic may split long no-blank-line sections.
    ConservativeHeuristic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SectionBoundaryKind {
    FileStart,
    Heading,
    SceneBreak,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedSpan {
    // Ordinal preserves document order independently of later persistence IDs.
    pub ordinal: usize,
    pub span_type: SpanType,
    pub text: String,
    // Normalized text is a retrieval/comparison sidecar; it never replaces the original source text.
    pub normalized_text: String,
    pub start_line: usize,
    pub end_line: usize,
    // Offsets are exclusive at the end boundary so later editor slices can use standard Rust ranges.
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedSection {
    pub ordinal: usize,
    pub text: String,
    // Section normalization keeps paragraph boundaries searchable without mutating the stored Markdown text.
    pub normalized_text: String,
    // Section boundary metadata lets later prompt assembly distinguish file-start context from heading/scene transitions.
    pub boundary_kind: SectionBoundaryKind,
    pub boundary_text: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedScene {
    pub ordinal: usize,
    pub text: String,
    // Scenes expose the same normalized sidecar so scene-level retrieval can compare across formatting differences.
    pub normalized_text: String,
    pub separator: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub start_span_ordinal: usize,
    pub end_span_ordinal: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedMarkdownDocument {
    pub spans: Vec<ParsedSpan>,
    pub sections: Vec<ParsedSection>,
    pub scenes: Vec<ParsedScene>,
}
