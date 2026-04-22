use writing_assist_core::{SentenceType, StructuralMarkerKind};
use writing_assist_index::{parse_markdown_document, preprocess_parsed_document};

#[test]
fn preprocessing_normalizes_punctuation_without_losing_token_offsets() {
    let parsed = parse_markdown_document("“Captain’s log…” — Kohaku said.\n");

    let preprocessed = preprocess_parsed_document(&parsed);
    let span = preprocessed
        .spans
        .iter()
        .find(|span| span.span_ordinal == 0)
        .expect("expected first preprocessed span");

    assert_eq!(preprocessed.tokenizer_version, "deterministic_v1");
    assert!(
        span.normalized_text
            .contains("\"Captain's log...\" - Kohaku said.")
    );

    let captain = span
        .tokens
        .iter()
        .find(|token| token.surface == "Captain’s")
        .expect("expected token for Captain’s");
    assert_eq!(captain.normalized, "Captain's");
    assert_eq!(captain.start_char, 1);
    assert_eq!(captain.end_char, 10);

    let dash = span
        .tokens
        .iter()
        .find(|token| token.normalized == "-")
        .expect("expected normalized dash token");
    assert_eq!(dash.surface, "—");
}

#[test]
fn preprocessing_extracts_quote_spans_and_sentence_boundaries() {
    let parsed =
        parse_markdown_document("“Kohaku, move,” Captain Mara said.\n\nThen she left the dock.\n");

    let preprocessed = preprocess_parsed_document(&parsed);
    let first_span = preprocessed
        .spans
        .iter()
        .find(|span| span.span_ordinal == 0)
        .expect("expected first preprocessed span");

    assert_eq!(first_span.quote_spans.len(), 1);
    assert_eq!(
        first_span.quote_spans[0].normalized_text,
        "\"Kohaku, move,\""
    );
    assert_eq!(first_span.sentences.len(), 1);
    assert_eq!(
        first_span.sentences[0].sentence_type,
        SentenceType::Dialogue
    );

    let second_span = preprocessed
        .spans
        .iter()
        .find(|span| span.span_ordinal == 1)
        .expect("expected second preprocessed span");
    assert_eq!(second_span.sentences.len(), 1);
    assert_eq!(
        second_span.sentences[0].sentence_type,
        SentenceType::Narrative
    );
}

#[test]
fn preprocessing_keeps_title_abbreviations_inside_the_same_sentence() {
    let parsed =
        parse_markdown_document("“I understand, Mrs. Yō.”\n\nDr. Earlean reviewed the chart.\n");

    let preprocessed = preprocess_parsed_document(&parsed);
    let first_span = preprocessed
        .spans
        .iter()
        .find(|span| span.span_ordinal == 0)
        .expect("expected first preprocessed span");
    let second_span = preprocessed
        .spans
        .iter()
        .find(|span| span.span_ordinal == 1)
        .expect("expected second preprocessed span");

    assert_eq!(first_span.sentences.len(), 1);
    assert!(first_span.sentences[0].text.contains("Mrs. Yō."));
    assert_eq!(second_span.sentences.len(), 1);
    assert_eq!(
        second_span.sentences[0].text,
        "Dr. Earlean reviewed the chart."
    );
}

#[test]
fn preprocessing_emits_heading_list_and_scene_break_markers() {
    let parsed =
        parse_markdown_document("# Plan\n\n- Bring Kohaku\n- Call Yō\n\n***\n\nNext scene.\n");

    let preprocessed = preprocess_parsed_document(&parsed);
    let marker_kinds: Vec<_> = preprocessed
        .structural_markers
        .iter()
        .map(|marker| marker.kind.clone())
        .collect();

    assert!(marker_kinds.contains(&StructuralMarkerKind::Heading));
    assert!(marker_kinds.contains(&StructuralMarkerKind::ListItem));
    assert!(marker_kinds.contains(&StructuralMarkerKind::SceneBreak));

    let list_item_count = preprocessed
        .structural_markers
        .iter()
        .filter(|marker| marker.kind == StructuralMarkerKind::ListItem)
        .count();
    assert_eq!(list_item_count, 2);
}
