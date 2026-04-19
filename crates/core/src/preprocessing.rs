use serde::{Deserialize, Serialize};

use crate::SentenceType;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StructuralMarkerKind {
    Heading,
    Paragraph,
    ListItem,
    SceneBreak,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreprocessedToken {
    pub surface: String,
    pub normalized: String,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreprocessedSentence {
    pub ordinal: usize,
    pub text: String,
    pub normalized_text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub sentence_type: SentenceType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreprocessedQuoteSpan {
    pub ordinal: usize,
    pub text: String,
    pub normalized_text: String,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreprocessedSpan {
    pub span_ordinal: usize,
    pub structural_kind: StructuralMarkerKind,
    pub normalized_text: String,
    pub tokens: Vec<PreprocessedToken>,
    pub sentences: Vec<PreprocessedSentence>,
    pub quote_spans: Vec<PreprocessedQuoteSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructuralMarker {
    pub kind: StructuralMarkerKind,
    pub span_ordinal: Option<usize>,
    pub section_ordinal: Option<usize>,
    pub text: Option<String>,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreprocessedDocument {
    pub tokenizer_version: String,
    pub spans: Vec<PreprocessedSpan>,
    pub structural_markers: Vec<StructuralMarker>,
}
