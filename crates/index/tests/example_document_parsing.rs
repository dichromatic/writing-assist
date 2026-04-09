use std::fs;
use std::path::PathBuf;

use writing_assist_core::SpanType;
use writing_assist_index::parse_markdown_document;

fn example_document_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/1. Radiant Firth.md")
}

#[test]
fn parses_real_example_document_without_empty_spans() {
    let markdown = fs::read_to_string(example_document_path()).expect("example document should load");
    let parsed = parse_markdown_document(&markdown);

    let heading_count = parsed
        .spans
        .iter()
        .filter(|span| span.span_type == SpanType::Heading)
        .count();
    let paragraph_count = parsed
        .spans
        .iter()
        .filter(|span| span.span_type == SpanType::Paragraph)
        .count();
    let scene_count = parsed
        .spans
        .iter()
        .filter(|span| span.span_type == SpanType::Scene)
        .count();

    // This fixture test is intended to show how the current parser behaves on a real manuscript file.
    println!(
        "fixture parse summary: spans={}, sections={}, headings={}, scenes={}, paragraphs={}, first_span={:?}",
        parsed.spans.len(),
        parsed.sections.len(),
        heading_count,
        scene_count,
        paragraph_count,
        parsed.spans.first().map(|span| &span.text)
    );

    assert!(!parsed.spans.is_empty(), "real documents should produce spans");
    assert!(
        !parsed.sections.is_empty(),
        "real documents should produce at least one section"
    );
    assert!(
        scene_count > 0,
        "the example manuscript uses thematic breaks, so the parser should emit scene spans"
    );
    assert!(
        parsed.sections.len() > 1,
        "scene breaks in the example manuscript should split the document into multiple sections"
    );
    assert!(
        parsed
            .spans
            .iter()
            .all(|span| !span.text.trim().is_empty()),
        "parser should not emit empty spans for real documents"
    );
    assert!(
        parsed
            .spans
            .iter()
            .enumerate()
            .all(|(index, span)| span.ordinal == index),
        "span ordinals should remain contiguous for real documents"
    );
}
