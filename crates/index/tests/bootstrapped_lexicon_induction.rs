use writing_assist_core::{
    BootstrappedLexiconEntryKind, LexiconSupportRecordKind, LexiconBootstrapRule,
};
use writing_assist_index::{
    cluster_document_mentions, compile_exact_phrase_lexicon_matcher,
    harvest_definition_candidates, harvest_exact_phrase_lexicon_mentions,
    harvest_mention_candidates, harvest_section_summary_seeds,
    harvest_structured_field_candidates, induce_bootstrapped_lexicon_entries,
    parse_markdown_document,
};
use writing_assist_core::DocumentArchetype;

#[test]
fn bootstraps_character_entries_from_seedless_dossier_evidence() {
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

    let entries = induce_bootstrapped_lexicon_entries(
        "story planning/harbor-profile.md",
        &clusters,
        &fields,
        &[],
    );

    let mara = entries
        .iter()
        .find(|entry| entry.canonical_surface == "Captain Mara")
        .expect("expected Captain Mara bootstrapped entry");

    assert_eq!(mara.kind, BootstrappedLexiconEntryKind::Character);
    assert!(mara.occurrence_count >= 2);
    assert!(mara
        .rule_sources
        .contains(&LexiconBootstrapRule::TitledMention));
    assert!(mara
        .rule_sources
        .contains(&LexiconBootstrapRule::AliasField));
    assert!(mara.evidence.iter().any(|evidence| {
        evidence.kind == LexiconSupportRecordKind::MentionCluster
    }));
    assert!(mara.evidence.iter().any(|evidence| {
        evidence.kind == LexiconSupportRecordKind::StructuredField
            && evidence.summary == "Alias: Mara"
    }));
}

#[test]
fn bootstraps_terminology_entries_from_definition_grounded_reference_evidence() {
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

    let entries = induce_bootstrapped_lexicon_entries(
        "world context/harmonics.md",
        &clusters,
        &[],
        &definitions,
    );

    let tau = entries
        .iter()
        .find(|entry| entry.canonical_surface == "Tau")
        .expect("expected Tau bootstrapped entry");

    assert_eq!(tau.kind, BootstrappedLexiconEntryKind::Terminology);
    assert!(tau
        .rule_sources
        .contains(&LexiconBootstrapRule::DefinitionTerm));
    assert!(tau.evidence.iter().any(|evidence| {
        evidence.kind == LexiconSupportRecordKind::Definition
            && evidence.summary.starts_with("Tau field =>")
    }));
}

#[test]
fn reference_bootstrapping_prefers_definition_grounded_terms_over_descriptive_fragments() {
    let parsed = parse_markdown_document(
        "# Harmonics\n\nSlipspace boundary: transition band between stable layers\n\nStrong gradients gather near the boundary.\n",
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

    let entries = induce_bootstrapped_lexicon_entries(
        "world context/harmonics.md",
        &clusters,
        &[],
        &definitions,
    );

    assert!(entries
        .iter()
        .any(|entry| entry.canonical_surface == "Slipspace boundary"
            && entry.kind == BootstrappedLexiconEntryKind::Terminology));
    assert!(!entries
        .iter()
        .any(|entry| entry.canonical_surface == "Strong"));
}

#[test]
fn planning_bootstrapping_prefers_field_grounded_entries_over_tone_vocabulary() {
    let parsed = parse_markdown_document(
        "# Briefing\n\nParticipants: Mara, Yori\nTone: Warm\nApproach: Precise\n\nMara briefs Yori before launch.\n",
    );

    let mentions = harvest_mention_candidates(
        "story planning/briefing.md",
        DocumentArchetype::StoryPlanning,
        &parsed,
    );
    let fields = harvest_structured_field_candidates(
        "story planning/briefing.md",
        DocumentArchetype::StoryPlanning,
        &parsed,
    );
    let seeds = harvest_section_summary_seeds(
        "story planning/briefing.md",
        DocumentArchetype::StoryPlanning,
        &parsed,
    );
    let clusters = cluster_document_mentions(
        "story planning/briefing.md",
        DocumentArchetype::StoryPlanning,
        &mentions,
        &fields,
        &[],
        &seeds,
    );

    let entries = induce_bootstrapped_lexicon_entries(
        "story planning/briefing.md",
        &clusters,
        &fields,
        &[],
    );

    assert!(entries
        .iter()
        .any(|entry| entry.canonical_surface == "Mara"
            && entry.kind == BootstrappedLexiconEntryKind::Character));
    assert!(entries
        .iter()
        .any(|entry| entry.canonical_surface == "Yori"
            && entry.kind == BootstrappedLexiconEntryKind::Character));
    assert!(!entries.iter().any(|entry| entry.canonical_surface == "Warm"));
    assert!(!entries
        .iter()
        .any(|entry| entry.canonical_surface == "Precise"));
}

#[test]
fn lexicon_matcher_recovers_lowercase_multiword_mentions_in_a_second_pass() {
    let lexicon_source = parse_markdown_document(
        "# Radiant Firth\n\nRole: Scout vessel\n\nRadiant Firth clears the harbor mouth.\n",
    );

    let source_mentions = harvest_mention_candidates(
        "story planning/fleet-notes.md",
        DocumentArchetype::DossierProfile,
        &lexicon_source,
    );
    let source_fields = harvest_structured_field_candidates(
        "story planning/fleet-notes.md",
        DocumentArchetype::DossierProfile,
        &lexicon_source,
    );
    let source_clusters = cluster_document_mentions(
        "story planning/fleet-notes.md",
        DocumentArchetype::DossierProfile,
        &source_mentions,
        &source_fields,
        &[],
        &[],
    );
    let entries = induce_bootstrapped_lexicon_entries(
        "story planning/fleet-notes.md",
        &source_clusters,
        &source_fields,
        &[],
    );
    let matcher = compile_exact_phrase_lexicon_matcher(&entries);

    let target = parse_markdown_document(
        "By dawn, the radiant firth had vanished into weather.\n",
    );

    let first_pass = harvest_mention_candidates(
        "chapters/chapter-lowercase.md",
        DocumentArchetype::Manuscript,
        &target,
    );
    let second_pass = harvest_exact_phrase_lexicon_mentions(
        "chapters/chapter-lowercase.md",
        DocumentArchetype::Manuscript,
        &target,
        &matcher,
    );

    assert!(!first_pass.iter().any(|candidate| candidate.surface == "radiant firth"));
    assert!(second_pass
        .iter()
        .any(|candidate| candidate.surface == "radiant firth"));
}

#[test]
fn lexicon_matcher_prefers_longest_overlapping_surface() {
    let parsed = parse_markdown_document(
        "# Radiant Firth\n\nRole: Scout vessel\n\nFirth weather logs remain archived.\n",
    );

    let mentions = harvest_mention_candidates(
        "story planning/fleet-notes.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );
    let fields = harvest_structured_field_candidates(
        "story planning/fleet-notes.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );
    let clusters = cluster_document_mentions(
        "story planning/fleet-notes.md",
        DocumentArchetype::DossierProfile,
        &mentions,
        &fields,
        &[],
        &[],
    );
    let entries = induce_bootstrapped_lexicon_entries(
        "story planning/fleet-notes.md",
        &clusters,
        &fields,
        &[],
    );
    let matcher = compile_exact_phrase_lexicon_matcher(&entries);

    let target = parse_markdown_document("The radiant firth drifted under blackout running.\n");
    let second_pass = harvest_exact_phrase_lexicon_mentions(
        "chapters/chapter-overlap.md",
        DocumentArchetype::Manuscript,
        &target,
        &matcher,
    );

    let surfaces: Vec<_> = second_pass.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(surfaces.contains(&"radiant firth"));
    assert!(!surfaces.contains(&"firth"));
}

#[test]
fn planning_exact_phrase_reuse_stays_field_grounded_instead_of_reusing_tone_words() {
    let source = parse_markdown_document(
        "# Briefing\n\nParticipants: Mara\nTone: Warm\n\nMara reviews the route.\n",
    );

    let source_mentions = harvest_mention_candidates(
        "story planning/briefing.md",
        DocumentArchetype::StoryPlanning,
        &source,
    );
    let source_fields = harvest_structured_field_candidates(
        "story planning/briefing.md",
        DocumentArchetype::StoryPlanning,
        &source,
    );
    let source_clusters = cluster_document_mentions(
        "story planning/briefing.md",
        DocumentArchetype::StoryPlanning,
        &source_mentions,
        &source_fields,
        &[],
        &[],
    );
    let entries = induce_bootstrapped_lexicon_entries(
        "story planning/briefing.md",
        &source_clusters,
        &source_fields,
        &[],
    );
    let matcher = compile_exact_phrase_lexicon_matcher(&entries);

    let target = parse_markdown_document("Later, mara waits while the warm corridor stays quiet.\n");
    let second_pass = harvest_exact_phrase_lexicon_mentions(
        "story planning/briefing-followup.md",
        DocumentArchetype::StoryPlanning,
        &target,
        &matcher,
    );

    let surfaces: Vec<_> = second_pass
        .iter()
        .map(|candidate| candidate.surface.as_str())
        .collect();

    assert!(surfaces.contains(&"mara"));
    assert!(!surfaces.contains(&"warm"));
}
