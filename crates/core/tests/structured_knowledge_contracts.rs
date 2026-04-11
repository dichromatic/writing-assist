use uuid::Uuid;
use writing_assist_core::{
    structured_knowledge_intended_use, DocumentArchetype, EntityProfileCandidate,
    MemoryReviewState, MemorySourceReference, MemoryStalenessState, StructuredKnowledgeCandidateKind,
    StructuredKnowledgeIntendedUse, TargetAnchor,
};

#[test]
fn document_archetypes_and_candidate_kinds_serialize_as_snake_case() {
    let archetype =
        serde_json::to_string(&DocumentArchetype::ExpositoryWorldArticle).expect("serialize archetype");
    let candidate_kind = serde_json::to_string(&StructuredKnowledgeCandidateKind::StoryArc)
        .expect("serialize candidate kind");
    let intended_use = serde_json::to_string(&StructuredKnowledgeIntendedUse::PlanningOnly)
        .expect("serialize intended use");

    assert_eq!(archetype, "\"expository_world_article\"");
    assert_eq!(candidate_kind, "\"story_arc\"");
    assert_eq!(intended_use, "\"planning_only\"");
}

#[test]
fn intended_use_defaults_follow_document_archetype() {
    assert_eq!(
        structured_knowledge_intended_use(
            DocumentArchetype::StoryPlanning,
            StructuredKnowledgeCandidateKind::StoryArc
        ),
        StructuredKnowledgeIntendedUse::PlanningOnly
    );
    assert_eq!(
        structured_knowledge_intended_use(
            DocumentArchetype::TaxonomyReference,
            StructuredKnowledgeCandidateKind::Terminology
        ),
        StructuredKnowledgeIntendedUse::CanonReference
    );
    assert_eq!(
        structured_knowledge_intended_use(
            DocumentArchetype::LooseNote,
            StructuredKnowledgeCandidateKind::ExtractiveSummary
        ),
        StructuredKnowledgeIntendedUse::WorkingContext
    );
}

#[test]
fn structured_candidate_records_preserve_source_links_and_review_gates() {
    let candidate = EntityProfileCandidate {
        id: Uuid::nil(),
        name: "Watanabe Yo".to_string(),
        role: Some("Captain".to_string()),
        traits: vec!["earnest".to_string(), "loyal".to_string()],
        details: vec!["Commands the Radiant Firth.".to_string()],
        source: MemorySourceReference::new(
            "story planning/estuary crew summaries.txt",
            vec![TargetAnchor::section(0)],
            10,
            120,
        ),
        intended_use: StructuredKnowledgeIntendedUse::CanonReference,
        review_state: MemoryReviewState::Pending,
        staleness_state: MemoryStalenessState::Current,
    };

    let serialized = serde_json::to_value(&candidate).expect("serialize entity profile");

    assert_eq!(serialized["source"]["document_path"], "story planning/estuary crew summaries.txt");
    assert_eq!(serialized["intended_use"], "canon_reference");
    assert_eq!(serialized["review_state"], "pending");
    assert_eq!(serialized["staleness_state"], "current");
  }
