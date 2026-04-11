use writing_assist_core::{DocumentArchetype, MentionClusterLinkKind, MentionFeature};
use writing_assist_index::{
    cluster_document_mentions, harvest_definition_candidates, harvest_mention_candidates,
    harvest_section_summary_seeds, harvest_structured_field_candidates, parse_markdown_document,
};

#[test]
fn clusters_titled_and_bare_mentions_and_links_local_fields_and_sections() {
    let parsed = parse_markdown_document(
        "# Captain Mara\n\nAlias: Mara\nRole: Harbormaster\n\nCaptain Mara signs the harbor ledger.\n\nMara confirms the tide window.\n",
    );

    let mentions = harvest_mention_candidates(
        "story planning/harbor-profile.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );
    let fields = harvest_structured_field_candidates(
        "story planning/harbor-profile.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );
    let seeds = harvest_section_summary_seeds(
        "story planning/harbor-profile.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );

    let clusters = cluster_document_mentions(
        "story planning/harbor-profile.md",
        DocumentArchetype::DossierProfile,
        &mentions,
        &fields,
        &[],
        &seeds,
    );

    let mara_cluster = clusters
        .iter()
        .find(|cluster| cluster.display_surface == "Captain Mara")
        .expect("expected Captain Mara cluster");

    assert!(mara_cluster.member_surfaces.contains(&"Captain Mara".to_string()));
    assert!(mara_cluster.member_surfaces.contains(&"Mara".to_string()));
    assert!(mara_cluster
        .aggregate_features
        .contains(&MentionFeature::Repeated));
    assert!(mara_cluster.linked_evidence.iter().any(|link| {
        link.kind == MentionClusterLinkKind::StructuredField && link.summary == "Alias: Mara"
    }));
    assert!(mara_cluster.linked_evidence.iter().any(|link| {
        link.kind == MentionClusterLinkKind::SectionSummarySeed && link.summary == "section:0"
    }));
}

#[test]
fn links_definition_evidence_to_matching_term_clusters() {
    let parsed = parse_markdown_document(
        "# Harmonics\n\nTau field: local resonance envelope\n\nTau field destabilizes near storm fronts.\n",
    );

    let mentions = harvest_mention_candidates(
        "world context/harmonics.md",
        DocumentArchetype::TaxonomyReference,
        &parsed,
    );
    let definitions = harvest_definition_candidates(
        "world context/harmonics.md",
        DocumentArchetype::TaxonomyReference,
        &parsed,
    );
    let seeds = harvest_section_summary_seeds(
        "world context/harmonics.md",
        DocumentArchetype::TaxonomyReference,
        &parsed,
    );

    let clusters = cluster_document_mentions(
        "world context/harmonics.md",
        DocumentArchetype::TaxonomyReference,
        &mentions,
        &[],
        &definitions,
        &seeds,
    );

    let tau_cluster = clusters
        .iter()
        .find(|cluster| cluster.display_surface == "Tau")
        .expect("expected Tau cluster");

    assert!(tau_cluster.linked_evidence.iter().any(|link| {
        link.kind == MentionClusterLinkKind::Definition
            && link.summary.starts_with("Tau field =>")
    }));
    assert!(tau_cluster.linked_evidence.iter().any(|link| {
        link.kind == MentionClusterLinkKind::SectionSummarySeed && link.summary == "section:0"
    }));
}
