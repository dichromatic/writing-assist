use uuid::Uuid;
use writing_assist_core::{
    DocumentType, MemoryReviewState, MemorySourceReference, MemoryStalenessState,
    ParsedMarkdownDocument, ReviewableFact, SpanType, TargetAnchor,
};

/// Extract conservative fact candidates from reference-style key/value lines.
///
/// This phase deliberately avoids broad prose inference. Facts only come from
/// reference documents and only when a line looks like structured glossary,
/// character sheet, or notes-style metadata.
pub fn extract_reviewable_facts(
    document_path: impl AsRef<str>,
    document_type: DocumentType,
    parsed: &ParsedMarkdownDocument,
) -> Vec<ReviewableFact> {
    if document_type != DocumentType::Reference {
        return Vec::new();
    }

    let document_path = document_path.as_ref();
    let mut facts = Vec::new();

    for span in parsed
        .spans
        .iter()
        .filter(|span| span.span_type == SpanType::Paragraph)
    {
        for line in span.text.lines() {
            let Some((subject, object)) = parse_reference_fact_line(line) else {
                continue;
            };

            facts.push(ReviewableFact::new(
                stable_fact_id(document_path, span.ordinal, &subject, &object),
                subject,
                "is",
                object,
                MemorySourceReference::new(
                    document_path,
                    vec![TargetAnchor::span(span.ordinal)],
                    span.start_char,
                    span.end_char,
                ),
                MemoryReviewState::Pending,
                MemoryStalenessState::Current,
            ));
        }
    }

    facts
}

fn parse_reference_fact_line(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    let trimmed = trimmed
        .trim_start_matches("- ")
        .trim_start_matches("* ")
        .trim_start_matches("+ ")
        .trim();

    let (subject, object) = if let Some((subject, object)) = trimmed.split_once(':') {
        (subject, object)
    } else if let Some((subject, object)) = trimmed.split_once(" - ") {
        (subject, object)
    } else if let Some((subject, object)) = trimmed.split_once(" — ") {
        (subject, object)
    } else {
        return None;
    };

    let subject = normalize_fact_part(subject);
    let object = normalize_fact_part(object);

    if subject.is_empty()
        || object.is_empty()
        || !subject.chars().any(|character| character.is_alphabetic())
        || !object.chars().any(|character| character.is_alphabetic())
    {
        return None;
    }

    Some((subject, object))
}

fn normalize_fact_part(text: &str) -> String {
    text.trim()
        .trim_matches(|character: char| matches!(character, ':' | '-' | '—'))
        .trim()
        .trim_end_matches('.')
        .trim()
        .to_string()
}

fn stable_fact_id(document_path: &str, span_ordinal: usize, subject: &str, object: &str) -> Uuid {
    let mut hash = 0xcbf29ce484222325_u128;

    for byte in document_path
        .bytes()
        .chain([0])
        .chain(span_ordinal.to_string().bytes())
        .chain([0])
        .chain(subject.bytes())
        .chain([0])
        .chain(object.bytes())
    {
        hash ^= byte as u128;
        hash = hash.wrapping_mul(0x00000100000001b3);
    }

    Uuid::from_u128(hash)
}
