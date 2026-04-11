use uuid::Uuid;
use writing_assist_core::{
    DocumentArchetype, EvidenceContext, MentionCandidate, MentionCluster,
    MentionClusterLink, MentionClusterLinkKind, MentionFeature, MentionOccurrence,
    MemorySourceReference, SentenceType, StructuredFieldCandidate, TargetAnchor,
};

#[test]
fn mention_features_serialize_as_snake_case() {
    let repeated =
        serde_json::to_string(&MentionFeature::HeadingMentioned).expect("serialize feature");
    let sentence_type =
        serde_json::to_string(&SentenceType::BlockQuote).expect("serialize sentence type");

    assert_eq!(repeated, "\"heading_mentioned\"");
    assert_eq!(sentence_type, "\"block_quote\"");
}

#[test]
fn evidence_candidates_preserve_source_links_and_contexts() {
    let mention = MentionCandidate {
        id: Uuid::nil(),
        surface: "Captain Mara".to_string(),
        normalized_surface: "captain mara".to_string(),
        source: MemorySourceReference::new(
            "chapters/chapter-1.md",
            vec![TargetAnchor::span(4), TargetAnchor::span(9)],
            10,
            42,
        ),
        occurrences: vec![MentionOccurrence {
            span_anchor: TargetAnchor::span(4),
            section_anchor: Some(TargetAnchor::section(0)),
            heading: Some("Arrival".to_string()),
            snippet: "Captain Mara reaches the harbor before dawn.".to_string(),
            sentence_type: SentenceType::Narrative,
            cooccurring_mentions: vec!["Radiant Firth".to_string()],
        }],
        aggregate_features: vec![MentionFeature::Repeated, MentionFeature::Titled],
        archetype: DocumentArchetype::Manuscript,
    };

    let serialized = serde_json::to_value(&mention).expect("serialize mention");

    assert_eq!(serialized["source"]["document_path"], "chapters/chapter-1.md");
    assert_eq!(serialized["occurrences"][0]["heading"], "Arrival");
    assert_eq!(serialized["occurrences"][0]["sentence_type"], "narrative");
    assert_eq!(serialized["aggregate_features"][0], "repeated");
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

#[test]
fn mention_clusters_preserve_members_and_structured_links() {
    let cluster = MentionCluster {
        id: Uuid::nil(),
        display_surface: "Captain Mara".to_string(),
        normalized_surface: "mara".to_string(),
        source: MemorySourceReference::new(
            "story planning/harbor-profile.txt",
            vec![TargetAnchor::span(0), TargetAnchor::span(2)],
            0,
            120,
        ),
        member_mention_ids: vec![Uuid::nil()],
        member_surfaces: vec!["Captain Mara".to_string(), "Mara".to_string()],
        occurrences: vec![MentionOccurrence {
            span_anchor: TargetAnchor::span(2),
            section_anchor: Some(TargetAnchor::section(0)),
            heading: Some("Captain Mara".to_string()),
            snippet: "Captain Mara signs the harbor ledger.".to_string(),
            sentence_type: SentenceType::Narrative,
            cooccurring_mentions: vec!["Radiant Firth".to_string()],
        }],
        aggregate_features: vec![MentionFeature::Repeated, MentionFeature::Titled],
        linked_evidence: vec![
            MentionClusterLink {
                kind: MentionClusterLinkKind::StructuredField,
                evidence_id: Uuid::nil(),
                summary: "Alias: Mara".to_string(),
            },
            MentionClusterLink {
                kind: MentionClusterLinkKind::SectionSummarySeed,
                evidence_id: Uuid::nil(),
                summary: "section:0".to_string(),
            },
        ],
        archetype: DocumentArchetype::DossierProfile,
    };

    let serialized = serde_json::to_value(&cluster).expect("serialize cluster");

    assert_eq!(serialized["display_surface"], "Captain Mara");
    assert_eq!(serialized["normalized_surface"], "mara");
    assert_eq!(serialized["member_surfaces"][1], "Mara");
    assert_eq!(serialized["linked_evidence"][0]["kind"], "structured_field");
    assert_eq!(serialized["linked_evidence"][1]["summary"], "section:0");
}
