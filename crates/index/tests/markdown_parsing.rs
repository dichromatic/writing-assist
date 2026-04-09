use writing_assist_core::SpanType;
use writing_assist_index::parse_markdown_document;

#[test]
fn headings_split_sections_correctly() {
    let parsed = parse_markdown_document(
        "# Opening\n\nFirst paragraph.\n\n## Next beat\n\nSecond paragraph.\n",
    );

    assert_eq!(parsed.sections.len(), 2);
    assert_eq!(parsed.sections[0].ordinal, 0);
    assert_eq!(parsed.sections[0].text, "# Opening\n\nFirst paragraph.");
    assert_eq!(parsed.sections[1].ordinal, 1);
    assert_eq!(parsed.sections[1].text, "## Next beat\n\nSecond paragraph.");

    assert_eq!(parsed.spans[0].span_type, SpanType::Heading);
    assert_eq!(parsed.spans[1].span_type, SpanType::Paragraph);
    assert_eq!(parsed.spans[2].span_type, SpanType::Heading);
    assert_eq!(parsed.spans[3].span_type, SpanType::Paragraph);
}

#[test]
fn extracts_paragraphs_across_blank_lines_without_creating_empty_spans() {
    let parsed = parse_markdown_document(
        "First paragraph line one.\nSecond line.\n\n\nThird paragraph.\n\n",
    );

    assert_eq!(parsed.spans.len(), 2);
    assert_eq!(parsed.spans[0].span_type, SpanType::Paragraph);
    assert_eq!(
        parsed.spans[0].text,
        "First paragraph line one.\nSecond line."
    );
    assert_eq!(parsed.spans[1].span_type, SpanType::Paragraph);
    assert_eq!(parsed.spans[1].text, "Third paragraph.");
}

#[test]
fn preserves_span_order_for_mixed_heading_and_paragraph_content() {
    let parsed = parse_markdown_document(
        "Intro paragraph.\n\n# Heading\n\nBody paragraph.\n\nAnother body paragraph.\n",
    );

    let ordered: Vec<_> = parsed
        .spans
        .iter()
        .map(|span| (span.ordinal, span.span_type.clone(), span.text.clone()))
        .collect();

    assert_eq!(
        ordered,
        vec![
            (0, SpanType::Paragraph, "Intro paragraph.".to_string()),
            (1, SpanType::Heading, "# Heading".to_string()),
            (2, SpanType::Paragraph, "Body paragraph.".to_string()),
            (3, SpanType::Paragraph, "Another body paragraph.".to_string()),
        ]
    );
}

#[test]
fn groups_content_before_first_heading_into_its_own_section() {
    let parsed = parse_markdown_document(
        "Intro paragraph.\n\n# Heading\n\nBody paragraph.\n",
    );

    assert_eq!(parsed.sections.len(), 2);
    assert_eq!(parsed.sections[0].text, "Intro paragraph.");
    assert_eq!(parsed.sections[1].text, "# Heading\n\nBody paragraph.");
}

#[test]
fn tracks_byte_and_char_offsets_for_spans_and_sections() {
    let parsed = parse_markdown_document("# Heading\n\nYō sails.\n");

    assert_eq!(parsed.spans[0].text, "# Heading");
    assert_eq!(parsed.spans[0].start_line, 0);
    assert_eq!(parsed.spans[0].end_line, 0);
    assert_eq!(parsed.spans[0].start_byte, 0);
    assert_eq!(parsed.spans[0].end_byte, 9);
    assert_eq!(parsed.spans[0].start_char, 0);
    assert_eq!(parsed.spans[0].end_char, 9);

    assert_eq!(parsed.spans[1].text, "Yō sails.");
    assert_eq!(parsed.spans[1].start_line, 2);
    assert_eq!(parsed.spans[1].end_line, 2);
    assert_eq!(parsed.spans[1].start_byte, 11);
    assert_eq!(parsed.spans[1].end_byte, 21);
    assert_eq!(parsed.spans[1].start_char, 11);
    assert_eq!(parsed.spans[1].end_char, 20);

    assert_eq!(parsed.sections[0].text, "# Heading\n\nYō sails.");
    assert_eq!(parsed.sections[0].start_line, 0);
    assert_eq!(parsed.sections[0].end_line, 2);
    assert_eq!(parsed.sections[0].start_byte, 0);
    assert_eq!(parsed.sections[0].end_byte, 21);
    assert_eq!(parsed.sections[0].start_char, 0);
    assert_eq!(parsed.sections[0].end_char, 20);
}

#[test]
fn thematic_breaks_become_scene_spans_and_split_sections() {
    let parsed = parse_markdown_document(
        "Opening paragraph.\n\n---\n\nNext scene paragraph.\n",
    );

    assert_eq!(parsed.scenes.len(), 2);
    assert_eq!(parsed.scenes[0].ordinal, 0);
    assert_eq!(parsed.scenes[0].text, "Opening paragraph.");
    assert_eq!(parsed.scenes[0].separator, None);
    assert_eq!(parsed.scenes[0].start_span_ordinal, 0);
    assert_eq!(parsed.scenes[0].end_span_ordinal, 0);

    assert_eq!(parsed.scenes[1].ordinal, 1);
    assert_eq!(parsed.scenes[1].text, "Next scene paragraph.");
    assert_eq!(parsed.scenes[1].separator, Some("---".to_string()));
    assert_eq!(parsed.scenes[1].start_span_ordinal, 2);
    assert_eq!(parsed.scenes[1].end_span_ordinal, 2);

    assert_eq!(parsed.sections.len(), 2);
    assert_eq!(parsed.sections[0].text, "Opening paragraph.");
    assert_eq!(parsed.sections[1].text, "Next scene paragraph.");

    assert_eq!(parsed.spans.len(), 3);
    assert_eq!(parsed.spans[0].span_type, SpanType::Paragraph);
    assert_eq!(parsed.spans[1].span_type, SpanType::Scene);
    assert_eq!(parsed.spans[1].text, "---");
    assert_eq!(parsed.spans[2].span_type, SpanType::Paragraph);
}

#[test]
fn applies_a_conservative_paragraph_heuristic_when_blank_lines_are_missing() {
    let parsed = parse_markdown_document(
        "First paragraph ends here.\nSecond paragraph starts here.\nThird paragraph starts here too.\n",
    );

    let paragraphs: Vec<_> = parsed
        .spans
        .iter()
        .filter(|span| span.span_type == SpanType::Paragraph)
        .map(|span| span.text.clone())
        .collect();

    assert_eq!(
        paragraphs,
        vec![
            "First paragraph ends here.".to_string(),
            "Second paragraph starts here.".to_string(),
            "Third paragraph starts here too.".to_string(),
        ]
    );
}

#[test]
fn headings_do_not_split_scenes() {
    let parsed = parse_markdown_document(
        "# Opening\n\nFirst paragraph.\n\n## Next beat\n\nSecond paragraph.\n",
    );

    assert_eq!(parsed.scenes.len(), 1);
    assert_eq!(parsed.scenes[0].text, "# Opening\n\nFirst paragraph.\n\n## Next beat\n\nSecond paragraph.");
    assert_eq!(parsed.scenes[0].separator, None);
    assert_eq!(parsed.scenes[0].start_span_ordinal, 0);
    assert_eq!(parsed.scenes[0].end_span_ordinal, 3);
}
