use uuid::Uuid;
use writing_assist_core::{
    MemoryReviewState, MemorySourceReference, MemoryStalenessState, ParsedMarkdownDocument,
    ParsedSection, ReviewableSummary, TargetAnchor,
};

const SUMMARY_TEXT_LIMIT: usize = 240;

/// Generate extractive summary candidates from existing section/document text.
///
/// These are deterministic scaffolds for the review queue, not provider-written
/// summaries. They stay pending until a later review step approves or replaces them.
pub fn generate_reviewable_summaries(
    document_path: impl AsRef<str>,
    parsed: &ParsedMarkdownDocument,
) -> Vec<ReviewableSummary> {
    let document_path = document_path.as_ref();
    let mut summaries = Vec::new();
    let section_summaries = parsed
        .sections
        .iter()
        .filter_map(|section| build_section_summary(document_path, section))
        .collect::<Vec<_>>();

    if let Some(document_summary) = build_document_summary(document_path, parsed) {
        summaries.push(document_summary);
    }

    if section_summaries.len() > 1 {
        summaries.extend(section_summaries);
    }

    summaries
}

fn build_document_summary(
    document_path: &str,
    parsed: &ParsedMarkdownDocument,
) -> Option<ReviewableSummary> {
    let summary_text = summarize_extractively(
        &parsed
            .sections
            .iter()
            .map(|section| section.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
    )?;
    let first_section = parsed.sections.first()?;
    let last_section = parsed.sections.last()?;
    let anchors = parsed
        .sections
        .iter()
        .map(|section| TargetAnchor::section(section.ordinal))
        .collect::<Vec<_>>();

    Some(ReviewableSummary::new(
        stable_summary_id(document_path, "document", &summary_text),
        "document",
        summary_text,
        MemorySourceReference::new(
            document_path,
            anchors,
            first_section.start_char,
            last_section.end_char,
        ),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    ))
}

fn build_section_summary(
    document_path: &str,
    section: &ParsedSection,
) -> Option<ReviewableSummary> {
    let summary_text = summarize_extractively(&section.text)?;
    let scope = format!("section:{}", section.ordinal);

    Some(ReviewableSummary::new(
        stable_summary_id(document_path, &scope, &summary_text),
        scope,
        summary_text,
        MemorySourceReference::new(
            document_path,
            vec![TargetAnchor::section(section.ordinal)],
            section.start_char,
            section.end_char,
        ),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    ))
}

fn summarize_extractively(text: &str) -> Option<String> {
    let mut summary_parts = Vec::new();
    let mut current_length = 0;

    for line in text.lines() {
        let line = line.trim();

        if line.is_empty() || is_scene_break(line) {
            continue;
        }

        let cleaned_line = strip_markdown_heading(line)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        if cleaned_line.is_empty() {
            continue;
        }

        let separator_length = if summary_parts.is_empty() { 0 } else { 1 };
        let remaining = SUMMARY_TEXT_LIMIT.saturating_sub(current_length + separator_length);

        if remaining == 0 {
            break;
        }

        let snippet = truncate_to_char_limit(&cleaned_line, remaining);

        if snippet.is_empty() {
            break;
        }

        current_length += separator_length + snippet.chars().count();
        summary_parts.push(snippet);

        if current_length >= SUMMARY_TEXT_LIMIT {
            break;
        }
    }

    if summary_parts.is_empty() {
        None
    } else {
        Some(summary_parts.join(" "))
    }
}

fn strip_markdown_heading(line: &str) -> &str {
    line.trim_start_matches('#').trim()
}

fn is_scene_break(line: &str) -> bool {
    matches!(line, "---" | "***" | "___")
}

fn truncate_to_char_limit(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

fn stable_summary_id(document_path: &str, scope: &str, text: &str) -> Uuid {
    let mut hash = 0xcbf29ce484222325_u128;

    for byte in document_path
        .bytes()
        .chain([0])
        .chain(scope.bytes())
        .chain([0])
        .chain(text.bytes())
    {
        hash ^= byte as u128;
        hash = hash.wrapping_mul(0x00000100000001b3);
    }

    Uuid::from_u128(hash)
}
