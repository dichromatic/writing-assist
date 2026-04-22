use uuid::Uuid;
use writing_assist_core::{
    DocumentArchetype, EvidenceSuppressionReason, MemorySourceReference, PromotedEvidenceBundle,
    PromotedEvidenceCandidate, PromotedEvidenceKind, PromotionReason, SuppressedEvidenceCandidate,
    TargetAnchor,
};

#[test]
fn promoted_evidence_bundle_serializes_source_linked_candidates_and_suppression_reasons() {
    let source = MemorySourceReference::new(
        "world context/harmonics.md",
        vec![TargetAnchor::span(4)],
        20,
        84,
    );

    let bundle = PromotedEvidenceBundle {
        promoted: vec![PromotedEvidenceCandidate {
            id: Uuid::nil(),
            display_surface: "Tau field".to_string(),
            normalized_surface: "tau field".to_string(),
            kind: PromotedEvidenceKind::Terminology,
            source: source.clone(),
            archetype: DocumentArchetype::TaxonomyReference,
            evidence_ids: vec![Uuid::nil()],
            reasons: vec![
                PromotionReason::DefinitionBacked,
                PromotionReason::LexiconBacked,
            ],
            snippets: vec!["Tau field: local resonance envelope".to_string()],
        }],
        review_only: vec![],
        suppressed: vec![SuppressedEvidenceCandidate {
            evidence_id: Uuid::nil(),
            display_surface: "SAP".to_string(),
            normalized_surface: "sap".to_string(),
            source,
            archetype: DocumentArchetype::TaxonomyReference,
            reason: EvidenceSuppressionReason::UnresolvedAbbreviation,
        }],
    };

    let serialized = serde_json::to_value(&bundle).expect("serialize promoted evidence bundle");

    assert_eq!(serialized["promoted"][0]["kind"], "terminology");
    assert_eq!(serialized["promoted"][0]["reasons"][0], "definition_backed");
    assert_eq!(
        serialized["suppressed"][0]["reason"],
        "unresolved_abbreviation"
    );
}
