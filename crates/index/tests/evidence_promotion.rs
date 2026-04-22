use uuid::Uuid;
use writing_assist_core::{
    DefinitionCandidate, DocumentArchetype, EvidenceContext, EvidenceSuppressionReason,
    MemorySourceReference, MentionCluster, MentionClusterLink, MentionClusterLinkKind,
    MentionFeature, MentionOccurrence, PromotedEvidenceKind, PromotionReason, SentenceType,
    TargetAnchor,
};
use writing_assist_index::{
    cluster_document_mentions, harvest_mention_candidates, harvest_section_summary_seeds,
    harvest_structured_field_candidates, induce_bootstrapped_lexicon_entries,
    parse_markdown_document, promote_evidence_for_context,
};

#[test]
fn promotes_strong_mentions_and_suppresses_weak_singletons_for_context_input() {
    let clusters = vec![
        cluster_with_features(
            "chapters/chapter.md",
            "Radiant Firth",
            DocumentArchetype::Manuscript,
            vec![MentionFeature::Repeated, MentionFeature::MultiWord],
            2,
            vec![],
        ),
        cluster_with_features(
            "chapters/chapter.md",
            "Warm",
            DocumentArchetype::Manuscript,
            vec![],
            1,
            vec![],
        ),
    ];

    let bundle = promote_evidence_for_context(&clusters, &[], &[], &[]);

    assert!(bundle.promoted.iter().any(|candidate| {
        candidate.display_surface == "Radiant Firth"
            && candidate.kind == PromotedEvidenceKind::EntityLike
            && candidate
                .reasons
                .contains(&PromotionReason::RepeatedMention)
    }));
    assert!(bundle.suppressed.iter().any(|candidate| {
        candidate.display_surface == "Warm"
            && candidate.reason == EvidenceSuppressionReason::WeakSingleton
    }));
}

#[test]
fn promotes_definition_backed_terms_but_suppresses_unresolved_abbreviations() {
    let definition = definition_candidate(
        "world context/harmonics.md",
        "Tau field",
        "local resonance envelope",
        20,
    );
    let clusters = vec![
        cluster_with_features(
            "world context/harmonics.md",
            "Tau field",
            DocumentArchetype::TaxonomyReference,
            vec![MentionFeature::MultiWord],
            1,
            vec![MentionClusterLink {
                kind: MentionClusterLinkKind::Definition,
                evidence_id: definition.id,
                summary: "Tau field => local resonance envelope".to_string(),
            }],
        ),
        cluster_with_features(
            "world context/harmonics.md",
            "SAP",
            DocumentArchetype::TaxonomyReference,
            vec![MentionFeature::Repeated],
            2,
            vec![],
        ),
    ];

    let bundle = promote_evidence_for_context(&clusters, &[], &[definition], &[]);

    assert!(bundle.promoted.iter().any(|candidate| {
        candidate.display_surface == "Tau field"
            && candidate.kind == PromotedEvidenceKind::Terminology
            && candidate
                .reasons
                .contains(&PromotionReason::DefinitionBacked)
    }));
    assert!(bundle.suppressed.iter().any(|candidate| {
        candidate.display_surface == "SAP"
            && candidate.reason == EvidenceSuppressionReason::UnresolvedAbbreviation
    }));
}

#[test]
fn keeps_relationship_shaped_fields_review_only_without_final_relationship_promotion() {
    let parsed = parse_markdown_document(
        "# Profile\n\nAlias: Mara\nRelationship: Yori mentor\n\nMara briefs Yori.\n",
    );

    let mentions = harvest_mention_candidates(
        "story planning/profile.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );
    let fields = harvest_structured_field_candidates(
        "story planning/profile.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );
    let seeds = harvest_section_summary_seeds(
        "story planning/profile.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );
    let clusters = cluster_document_mentions(
        "story planning/profile.md",
        DocumentArchetype::DossierProfile,
        &mentions,
        &fields,
        &[],
        &seeds,
    );
    let entries =
        induce_bootstrapped_lexicon_entries("story planning/profile.md", &clusters, &fields, &[]);

    let bundle = promote_evidence_for_context(&clusters, &fields, &[], &entries);

    assert!(bundle.review_only.iter().any(|candidate| {
        candidate.display_surface == "Relationship: Yori mentor"
            && candidate.kind == PromotedEvidenceKind::FieldBackedContext
    }));
    assert!(
        bundle
            .promoted
            .iter()
            .all(|candidate| { candidate.display_surface != "Relationship: Yori mentor" })
    );
}

fn cluster_with_features(
    document_path: &str,
    surface: &str,
    archetype: DocumentArchetype,
    aggregate_features: Vec<MentionFeature>,
    occurrence_count: usize,
    linked_evidence: Vec<MentionClusterLink>,
) -> MentionCluster {
    let normalized_surface = normalize_surface(surface);
    let occurrences = (0..occurrence_count)
        .map(|index| MentionOccurrence {
            span_anchor: TargetAnchor::span(index),
            section_anchor: Some(TargetAnchor::section(0)),
            heading: None,
            snippet: format!("{surface} appears in context."),
            sentence_type: SentenceType::Narrative,
            cooccurring_mentions: Vec::new(),
        })
        .collect::<Vec<_>>();

    MentionCluster {
        id: Uuid::from_u128(surface.bytes().fold(1_u128, |hash, byte| {
            hash.wrapping_mul(31).wrapping_add(byte as u128)
        })),
        display_surface: surface.to_string(),
        normalized_surface,
        source: MemorySourceReference::new(
            document_path,
            vec![TargetAnchor::span(0)],
            0,
            surface.len(),
        ),
        member_mention_ids: Vec::new(),
        member_surfaces: vec![surface.to_string()],
        occurrences,
        aggregate_features,
        linked_evidence,
        archetype,
    }
}

fn definition_candidate(
    document_path: &str,
    term: &str,
    definition: &str,
    id_seed: u128,
) -> DefinitionCandidate {
    DefinitionCandidate {
        id: Uuid::from_u128(id_seed),
        term: term.to_string(),
        definition: definition.to_string(),
        source: MemorySourceReference::new(
            document_path,
            vec![TargetAnchor::span(0)],
            0,
            term.len() + definition.len(),
        ),
        contexts: vec![EvidenceContext {
            span_anchor: TargetAnchor::span(0),
            section_anchor: Some(TargetAnchor::section(0)),
            heading: None,
            excerpt: format!("{term}: {definition}"),
        }],
        archetype: DocumentArchetype::TaxonomyReference,
    }
}

fn normalize_surface(surface: &str) -> String {
    surface
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
