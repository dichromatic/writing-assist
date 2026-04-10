use uuid::Uuid;
use writing_assist_core::{
    EntityCandidate, MemoryReviewState, MemorySourceReference, MemoryStalenessState,
    ReviewableFact, ReviewableSummary, TargetAnchor,
};

fn source_reference() -> MemorySourceReference {
    MemorySourceReference::new(
        "chapters/chapter-1.md",
        vec![TargetAnchor::span(7), TargetAnchor::section(2)],
        120,
        180,
    )
}

#[test]
fn reviewable_memory_is_reusable_only_when_approved_and_current() {
    let approved = EntityCandidate::new(
        Uuid::from_u128(1),
        "Radiant Firth",
        source_reference(),
        MemoryReviewState::Approved,
        MemoryStalenessState::Current,
    );
    let pending = EntityCandidate::new(
        Uuid::from_u128(2),
        "Radiant Firth",
        source_reference(),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    );
    let rejected = EntityCandidate::new(
        Uuid::from_u128(3),
        "Radiant Firth",
        source_reference(),
        MemoryReviewState::Rejected,
        MemoryStalenessState::Current,
    );
    let stale = EntityCandidate::new(
        Uuid::from_u128(4),
        "Radiant Firth",
        source_reference(),
        MemoryReviewState::Approved,
        MemoryStalenessState::Stale,
    );

    assert!(approved.is_reusable());
    assert!(!pending.is_reusable());
    assert!(!rejected.is_reusable());
    assert!(!stale.is_reusable());
}

#[test]
fn approved_fact_preserves_source_document_and_span_anchors() {
    let fact = ReviewableFact::new(
        Uuid::from_u128(11),
        "Radiant Firth",
        "located_in",
        "Outer shipping lane",
        source_reference(),
        MemoryReviewState::Approved,
        MemoryStalenessState::Current,
    );

    assert_eq!(fact.source.document_path, "chapters/chapter-1.md");
    assert_eq!(fact.source.anchors, vec![TargetAnchor::span(7), TargetAnchor::section(2)]);
    assert!(fact.is_reusable());
}

#[test]
fn memory_ids_are_stable_fields_not_inferred_from_display_text() {
    let first = ReviewableSummary::new(
        Uuid::from_u128(21),
        "chapter",
        "The crew reaches the Radiant Firth.",
        source_reference(),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    );
    let second = ReviewableSummary::new(
        Uuid::from_u128(22),
        "chapter",
        "The crew reaches the Radiant Firth.",
        source_reference(),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    );

    assert_ne!(first.id, second.id);
    assert_eq!(first.text, second.text);
}

#[test]
fn review_and_staleness_states_serialize_as_snake_case() {
    assert_eq!(
        serde_json::to_string(&MemoryReviewState::Pending).expect("serialize review state"),
        r#""pending""#
    );
    assert_eq!(
        serde_json::to_string(&MemoryReviewState::Approved).expect("serialize review state"),
        r#""approved""#
    );
    assert_eq!(
        serde_json::to_string(&MemoryReviewState::Rejected).expect("serialize review state"),
        r#""rejected""#
    );
    assert_eq!(
        serde_json::to_string(&MemoryStalenessState::Current).expect("serialize stale state"),
        r#""current""#
    );
    assert_eq!(
        serde_json::to_string(&MemoryStalenessState::Stale).expect("serialize stale state"),
        r#""stale""#
    );
}
