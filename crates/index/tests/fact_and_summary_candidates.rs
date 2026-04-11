use writing_assist_core::{
    DocumentType, MemoryReviewState, MemoryStalenessState, TargetAnchor,
};
use writing_assist_index::{
    extract_reviewable_facts, generate_reviewable_summaries, parse_markdown_document,
};

#[test]
fn extracts_pending_facts_from_structured_reference_lines() {
    let parsed = parse_markdown_document(
        "Captain Mara: First watch officer\n\nRadiant Firth: Survey cutter\n",
    );

    let facts =
        extract_reviewable_facts("reference/crew-sheet.md", DocumentType::Reference, &parsed);

    assert_eq!(facts.len(), 2);
    assert_eq!(facts[0].subject, "Captain Mara");
    assert_eq!(facts[0].predicate, "is");
    assert_eq!(facts[0].object, "First watch officer");
    assert_eq!(facts[0].review_state, MemoryReviewState::Pending);
    assert_eq!(facts[0].staleness_state, MemoryStalenessState::Current);
    assert_eq!(facts[0].source.document_path, "reference/crew-sheet.md");
    assert_eq!(facts[0].source.anchors, vec![TargetAnchor::span(0)]);

    assert_eq!(facts[1].subject, "Radiant Firth");
    assert_eq!(facts[1].predicate, "is");
    assert_eq!(facts[1].object, "Survey cutter");
    assert_eq!(facts[1].source.anchors, vec![TargetAnchor::span(1)]);
}

#[test]
fn does_not_extract_facts_from_manuscript_documents() {
    let parsed = parse_markdown_document("Captain Mara: wait for the tide.\n");

    let facts =
        extract_reviewable_facts("chapters/chapter-1.md", DocumentType::Manuscript, &parsed);

    assert!(facts.is_empty());
}

#[test]
fn generates_pending_document_and_section_summaries_with_section_anchors() {
    let parsed = parse_markdown_document(
        "# Arrival\n\nCaptain Mara reaches the harbor before dawn.\n\n# Departure\n\nRadiant Firth leaves with the tide.\n",
    );

    let summaries = generate_reviewable_summaries("chapters/chapter-1.md", &parsed);

    assert_eq!(summaries.len(), 3);

    let document_summary = summaries
        .iter()
        .find(|summary| summary.scope == "document")
        .expect("expected document summary");
    assert_eq!(document_summary.review_state, MemoryReviewState::Pending);
    assert_eq!(document_summary.staleness_state, MemoryStalenessState::Current);
    assert_eq!(document_summary.source.document_path, "chapters/chapter-1.md");
    assert_eq!(
        document_summary.source.anchors,
        vec![TargetAnchor::section(0), TargetAnchor::section(1)]
    );
    assert!(document_summary.text.contains("Arrival"));
    assert!(document_summary.text.contains("Captain Mara reaches the harbor before dawn."));

    let first_section_summary = summaries
        .iter()
        .find(|summary| summary.scope == "section:0")
        .expect("expected first section summary");
    assert_eq!(
        first_section_summary.source.anchors,
        vec![TargetAnchor::section(0)]
    );
    assert!(first_section_summary.text.contains("Arrival"));

    let second_section_summary = summaries
        .iter()
        .find(|summary| summary.scope == "section:1")
        .expect("expected second section summary");
    assert_eq!(
        second_section_summary.source.anchors,
        vec![TargetAnchor::section(1)]
    );
    assert!(second_section_summary.text.contains("Departure"));
}

#[test]
fn summary_generation_is_deterministic_and_skips_empty_text() {
    let parsed = parse_markdown_document("---\n\n");

    let first = generate_reviewable_summaries("chapters/chapter-2.md", &parsed);
    let second = generate_reviewable_summaries("chapters/chapter-2.md", &parsed);

    assert!(first.is_empty());
    assert_eq!(first, second);
}
