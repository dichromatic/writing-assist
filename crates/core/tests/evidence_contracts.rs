use uuid::Uuid;
use writing_assist_core::{
    DocumentArchetype, EvidenceContext, MentionCandidate, MentionFeature,
    MemorySourceReference, StructuredFieldCandidate, TargetAnchor,
};

#[test]
fn mention_features_serialize_as_snake_case() {
    let repeated =
        serde_json::to_string(&MentionFeature::HeadingMentioned).expect("serialize feature");

    assert_eq!(repeated, "\"heading_mentioned\"");
}

#[test]
fn evidence_candidates_preserve_source_links_and_contexts() {
    let mention = MentionCandidate {
        id: Uuid::nil(),
        surface: "Captain Mara".to_string(),
        normalized_surface: "captain mara".to_string(),
        occurrence_count: 2,
        source: MemorySourceReference::new(
            "chapters/chapter-1.md",
            vec![TargetAnchor::span(4), TargetAnchor::span(9)],
            10,
            42,
        ),
        contexts: vec![EvidenceContext {
            span_anchor: TargetAnchor::span(4),
            section_anchor: Some(TargetAnchor::section(0)),
            heading: Some("Arrival".to_string()),
            excerpt: "Captain Mara reaches the harbor before dawn.".to_string(),
        }],
        features: vec![MentionFeature::Repeated, MentionFeature::Titled],
        archetype: DocumentArchetype::Manuscript,
    };

    let serialized = serde_json::to_value(&mention).expect("serialize mention");

    assert_eq!(serialized["source"]["document_path"], "chapters/chapter-1.md");
    assert_eq!(serialized["contexts"][0]["heading"], "Arrival");
    assert_eq!(serialized["features"][0], "repeated");
    assert_eq!(serialized["archetype"], "manuscript");
}

#[test]
fn structured_field_candidates_are_provider_agnostic_evidence_records() {
    let field = StructuredFieldCandidate {
        id: Uuid::nil(),
        label: "Role".to_string(),
        value: "Operations Lead".to_string(),
        source: MemorySourceReference::new(
            "story planning/crew-profiles.txt",
            vec![TargetAnchor::span(2)],
            25,
            48,
        ),
        contexts: vec![EvidenceContext {
            span_anchor: TargetAnchor::span(2),
            section_anchor: Some(TargetAnchor::section(0)),
            heading: Some("ALINA VOSS".to_string()),
            excerpt: "Role: Operations Lead".to_string(),
        }],
        archetype: DocumentArchetype::DossierProfile,
    };

    let serialized = serde_json::to_value(&field).expect("serialize field");

    assert_eq!(serialized["label"], "Role");
    assert_eq!(serialized["value"], "Operations Lead");
    assert_eq!(serialized["archetype"], "dossier_profile");
}
