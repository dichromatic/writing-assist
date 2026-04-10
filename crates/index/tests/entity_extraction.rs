use writing_assist_core::{MemoryReviewState, MemoryStalenessState, TargetAnchor};
use writing_assist_index::{extract_entity_candidates, parse_markdown_document};

#[test]
fn extracts_pending_entities_with_source_span_anchors() {
    let parsed = parse_markdown_document(
        "The Radiant Firth crossed the outer lane.\n\nRadiant Firth returned after dusk.\n",
    );

    let candidates = extract_entity_candidates("chapters/chapter-1.md", &parsed);

    let radiant_firth = candidates
        .iter()
        .find(|candidate| candidate.name == "Radiant Firth")
        .expect("expected repeated ship name candidate");

    assert_eq!(radiant_firth.review_state, MemoryReviewState::Pending);
    assert_eq!(radiant_firth.staleness_state, MemoryStalenessState::Current);
    assert_eq!(radiant_firth.source.document_path, "chapters/chapter-1.md");
    assert_eq!(
        radiant_firth.source.anchors,
        vec![TargetAnchor::span(0), TargetAnchor::span(1)]
    );
}

#[test]
fn deduplicates_repeated_entities_deterministically() {
    let parsed = parse_markdown_document(
        "Ysolde watches the harbor.\n\nYsolde records the omen.\n\nRadiant Firth waits below.\n",
    );

    let first = extract_entity_candidates("chapters/chapter-1.md", &parsed);
    let second = extract_entity_candidates("chapters/chapter-1.md", &parsed);

    let names: Vec<_> = first.iter().map(|candidate| candidate.name.as_str()).collect();

    assert_eq!(names, vec!["Ysolde", "Radiant Firth"]);
    assert_eq!(first, second);
}

#[test]
fn filters_ordinary_sentence_initial_single_use_words() {
    let parsed = parse_markdown_document(
        "Opening lines describe the harbor.\n\nAnother sentence mentions nothing named.\n\n\
        Captain Mara enters.\n",
    );

    let candidates = extract_entity_candidates("chapters/chapter-1.md", &parsed);
    let names: Vec<_> = candidates.iter().map(|candidate| candidate.name.as_str()).collect();

    assert_eq!(names, vec!["Captain Mara"]);
}

#[test]
fn extraction_order_follows_first_mention_in_document_order() {
    let parsed = parse_markdown_document(
        "Captain Mara sees Ysolde.\n\nYsolde asks Mara about Radiant Firth.\n\n\
        Mara boards Radiant Firth.\n",
    );

    let candidates = extract_entity_candidates("chapters/chapter-1.md", &parsed);
    let names: Vec<_> = candidates.iter().map(|candidate| candidate.name.as_str()).collect();

    assert_eq!(
        names,
        vec!["Captain Mara", "Ysolde", "Mara", "Radiant Firth"]
    );
}
