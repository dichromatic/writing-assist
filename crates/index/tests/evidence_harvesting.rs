use writing_assist_core::{
    DocumentArchetype, MentionFeature, SentenceType, TargetAnchor,
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
    assert_eq!(kohaku.occurrences.len(), 2);
    assert_eq!(kohaku.source.document_path, "chapters/chapter-1.md");
    assert!(kohaku.aggregate_features.contains(&MentionFeature::PossessiveObserved));
    assert_eq!(
        kohaku.occurrences[0].section_anchor,
        Some(TargetAnchor::section(0))
    );
    assert_eq!(kohaku.occurrences[0].heading.as_deref(), Some("Arrival"));
    assert_eq!(kohaku.occurrences[0].sentence_type, SentenceType::Narrative);
    assert!(kohaku.occurrences[0].snippet.contains("Kohaku"));
    assert!(kohaku.occurrences[0]
        .cooccurring_mentions
        .contains(&"Captain Mara".to_string()));
}

#[test]
fn labels_dialogue_occurrences_so_the_semantic_layer_can_treat_them_differently() {
    let parsed = parse_markdown_document(
        "“Kohaku, move,” Captain Mara said.\n\nCaptain Mara entered the dock office.\n",
    );

    let mentions = harvest_mention_candidates(
        "chapters/chapter-2.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let captain_mara = mentions
        .iter()
        .find(|candidate| candidate.surface == "Captain Mara")
        .expect("expected titled mention");

    assert_eq!(captain_mara.occurrences.len(), 2);
    assert_eq!(captain_mara.occurrences[0].sentence_type, SentenceType::Dialogue);
    assert_eq!(captain_mara.occurrences[1].sentence_type, SentenceType::Narrative);
}

#[test]
fn manuscript_harvesting_does_not_carry_mentions_across_dialogue_punctuation() {
    let parsed = parse_markdown_document(
        "“Kohaku! Get me the repair drone now.”\n\nThe letter begins: Pathetic. Grotesque. How was she supposed to write it?\n",
    );

    let mentions = harvest_mention_candidates(
        "chapters/chapter-3.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"Kohaku Get"));
    assert!(!surfaces.contains(&"Pathetic Grotesque How"));
}

#[test]
fn manuscript_harvesting_keeps_abbreviated_titles_with_following_name() {
    let parsed = parse_markdown_document(
        "“I understand, Mrs. Yō.”\n\nDr. Earlean reviewed the chart.\n",
    );

    let mentions = harvest_mention_candidates(
        "chapters/chapter-4.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(surfaces.contains(&"Mrs Yō"));
    assert!(surfaces.contains(&"Dr Earlean"));
    assert!(!surfaces.contains(&"Mrs"));
    assert!(!surfaces.contains(&"Dr"));
}

#[test]
fn manuscript_harvesting_suppresses_repeated_dialogue_artifact_singletons() {
    let parsed = parse_markdown_document(
        "“What do we do now?”\n\n“What do we tell her?”\n\n“Let me think.”\n\n“Did you hear that?”\n",
    );

    let mentions = harvest_mention_candidates(
        "chapters/chapter-5.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"What"));
    assert!(!surfaces.contains(&"Let"));
    assert!(!surfaces.contains(&"Did"));
}

#[test]
fn manuscript_harvesting_suppresses_stutter_artifacts() {
    let parsed =
        parse_markdown_document("“S-sorry.”\n\n“I-I don’t know.”\n\n“Y-Yoshiko-chan...”\n");

    let mentions = harvest_mention_candidates(
        "chapters/chapter-6.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"S-sorry"));
    assert!(!surfaces.contains(&"I-I"));
    assert!(!surfaces.contains(&"Y-Yoshiko-chan"));
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
fn story_planning_harvesting_promotes_participant_fields_even_when_lowercase() {
    let parsed = parse_markdown_document(
        "participants: yō, kohaku, dia\n\nfocus: yoshiko\n\noutcome: the briefing lands cleanly\n",
    );

    let mentions = harvest_mention_candidates(
        "story planning/briefing-outline.txt",
        DocumentArchetype::StoryPlanning,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(surfaces.contains(&"yō"));
    assert!(surfaces.contains(&"kohaku"));
    assert!(surfaces.contains(&"dia"));
    assert!(surfaces.contains(&"yoshiko"));
    assert!(!surfaces.contains(&"the briefing lands cleanly"));
}

#[test]
fn taxonomy_reference_harvesting_promotes_definition_terms_into_mentions() {
    let parsed = parse_markdown_document(
        "tau field: local resonance envelope\n\nharmonic baseline = local magionic resonance floor\n",
    );

    let mentions = harvest_mention_candidates(
        "world context/terms.txt",
        DocumentArchetype::TaxonomyReference,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(surfaces.contains(&"tau field"));
    assert!(surfaces.contains(&"harmonic baseline"));
}

#[test]
fn dossier_profile_harvesting_promotes_alias_fields_even_when_lowercase() {
    let parsed = parse_markdown_document("# Captain Mara\n\nAlias: mara\n\nRole: harbormaster\n");

    let mentions = harvest_mention_candidates(
        "story planning/harbor-profile.md",
        DocumentArchetype::DossierProfile,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(surfaces.contains(&"Captain Mara"));
    assert!(surfaces.contains(&"mara"));
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
