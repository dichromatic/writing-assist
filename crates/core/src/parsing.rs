use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanType {
    Heading,
    Paragraph,
    Section,
    Window,
    Scene,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedSpan {
    // Ordinal preserves document order independently of later persistence IDs.
    pub ordinal: usize,
    pub span_type: SpanType,
    pub text: String,
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
