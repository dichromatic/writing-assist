use writing_assist_core::{
    DocumentArchetype, MentionFeature, TargetAnchor,
};
use writing_assist_index::{
    harvest_definition_candidates, harvest_mention_candidates, harvest_section_summary_seeds,
    harvest_structured_field_candidates, parse_markdown_document,
};

#[test]
fn harvests_manuscript_mentions_with_context_and_noise_suppression() {
    let parsed = parse_markdown_document(
        "# Arrival\n\nHey Kohaku, Captain Mara called to Radiant Firth.\n\nKohaku’s reply reached Captain Mara after dusk.\n",
    );

    let mentions = harvest_mention_candidates(
        "chapters/chapter-1.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(surfaces.contains(&"Kohaku"));
    assert!(surfaces.contains(&"Captain Mara"));
    assert!(surfaces.contains(&"Radiant Firth"));
    assert!(!surfaces.contains(&"Hey Kohaku"));
    assert!(!surfaces.contains(&"I"));

    let kohaku = mentions
        .iter()
        .find(|candidate| candidate.surface == "Kohaku")
        .expect("expected normalized possessive mention");
    assert_eq!(kohaku.occurrence_count, 2);
    assert_eq!(kohaku.source.document_path, "chapters/chapter-1.md");
    assert!(kohaku.features.contains(&MentionFeature::PossessiveObserved));
    assert_eq!(kohaku.contexts[0].section_anchor, Some(TargetAnchor::section(0)));
    assert_eq!(kohaku.contexts[0].heading.as_deref(), Some("Arrival"));
}

#[test]
fn harvests_structured_fields_from_profile_and_planning_lines() {
    let parsed = parse_markdown_document(
        "Role: Operations Lead\n\nOutcome: the crew reaches the station.\n",
    );

    let dossier_fields = harvest_structured_field_candidates(
        "story planning/crew-profile.txt",
        DocumentArchetype::DossierProfile,
        &parsed,
    );
    let planning_fields = harvest_structured_field_candidates(
        "story planning/outline.txt",
        DocumentArchetype::StoryPlanning,
        &parsed,
    );

    assert_eq!(dossier_fields.len(), 2);
    assert_eq!(planning_fields.len(), 2);
    assert_eq!(dossier_fields[0].label, "Role");
    assert_eq!(dossier_fields[0].value, "Operations Lead");
    assert_eq!(planning_fields[1].label, "Outcome");
}

#[test]
fn harvests_definition_candidates_for_taxonomy_references() {
    let parsed = parse_markdown_document(
        "Slipspace: superluminal transit through negative-polarity curvature.\n\n\
        Harmonic baseline = the local magionic resonance floor.\n",
    );

    let definitions = harvest_definition_candidates(
        "world context/terms.txt",
        DocumentArchetype::TaxonomyReference,
        &parsed,
    );

    assert_eq!(definitions.len(), 2);
    assert_eq!(definitions[0].term, "Slipspace");
    assert!(definitions[0]
        .definition
        .contains("superluminal transit through negative-polarity curvature"));
    assert_eq!(definitions[0].source.document_path, "world context/terms.txt");
}

#[test]
fn harvests_section_summary_seeds_with_heading_context() {
    let parsed = parse_markdown_document(
        "# Arrival\n\nCaptain Mara reaches the harbor before dawn.\n\n# Departure\n\nRadiant Firth leaves with the tide.\n",
    );

    let seeds = harvest_section_summary_seeds(
        "chapters/chapter-1.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    assert_eq!(seeds.len(), 2);
    assert_eq!(seeds[0].scope, "section:0");
    assert_eq!(seeds[0].contexts[0].heading.as_deref(), Some("Arrival"));
    assert!(seeds[0].text.contains("Captain Mara reaches the harbor before dawn."));
    assert_eq!(seeds[1].contexts[0].heading.as_deref(), Some("Departure"));
}
