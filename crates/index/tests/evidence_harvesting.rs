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
fn manuscript_harvesting_suppresses_remaining_repeated_singleton_noise() {
    let parsed = parse_markdown_document(
        "“Wait, I can explain.”\n\n\
         “Of course! Wait, sorry, that came out strangely.”\n\n\
         Since the hatch was still warm, she stepped back.\n\n\
         “Since we still have time, let’s keep moving.”\n\n\
         “I-I can prepare the report.”\n\n\
         “I-I don’t know.”\n",
    );

    let mentions = harvest_mention_candidates(
        "chapters/chapter-5b.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"Wait"));
    assert!(!surfaces.contains(&"Since"));
    assert!(!surfaces.contains(&"I-I"));
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
fn harvesting_suppresses_emoji_bearing_surfaces() {
    let manuscript = parse_markdown_document(
        "Kohaku🙂 reached the docking arm.\n\nCaptain Mara✨ checked the chart.\n",
    );
    let story_planning = parse_markdown_document(
        "participants: yō, kohaku🙂, dia\n\nfocus: yoshiko✨\n",
    );
    let taxonomy = parse_markdown_document(
        "tau🙂 field: local resonance envelope\n\nharmonic baseline: steady floor\n",
    );

    let manuscript_mentions = harvest_mention_candidates(
        "chapters/chapter-emoji.md",
        DocumentArchetype::Manuscript,
        &manuscript,
    );
    let planning_mentions = harvest_mention_candidates(
        "story planning/emoji-outline.txt",
        DocumentArchetype::StoryPlanning,
        &story_planning,
    );
    let taxonomy_mentions = harvest_mention_candidates(
        "world context/emoji-taxonomy.txt",
        DocumentArchetype::TaxonomyReference,
        &taxonomy,
    );

    let manuscript_surfaces: Vec<_> = manuscript_mentions
        .iter()
        .map(|candidate| candidate.surface.as_str())
        .collect();
    let planning_surfaces: Vec<_> = planning_mentions
        .iter()
        .map(|candidate| candidate.surface.as_str())
        .collect();
    let taxonomy_surfaces: Vec<_> = taxonomy_mentions
        .iter()
        .map(|candidate| candidate.surface.as_str())
        .collect();

    assert!(!manuscript_surfaces.contains(&"Kohaku🙂"));
    assert!(!manuscript_surfaces.contains(&"Captain Mara✨"));
    assert!(!planning_surfaces.contains(&"kohaku🙂"));
    assert!(!planning_surfaces.contains(&"yoshiko✨"));
    assert!(!taxonomy_surfaces.contains(&"tau🙂 field"));
}

#[test]
fn manuscript_harvesting_does_not_merge_quote_end_mentions_with_following_speaker_names() {
    let parsed = parse_markdown_document(
        "“I know what day it is, Kohaku,” Yō says.\n\n\
         “I... see. My apologies, Mrs. Yō.” Kohaku does not finish the thought.\n\n\
         “I’ll leave you to Kohaku then, Miss Yoshiko.” Earlean consults her tablet.\n",
    );

    let mentions = harvest_mention_candidates(
        "chapters/chapter-7.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"Kohaku Yō"));
    assert!(!surfaces.contains(&"Mrs Yō Kohaku"));
    assert!(!surfaces.contains(&"Miss Yoshiko Earlean"));
    assert!(surfaces.contains(&"Kohaku"));
    assert!(surfaces.contains(&"Mrs Yō"));
    assert!(surfaces.contains(&"Miss Yoshiko"));
}

#[test]
fn manuscript_harvesting_drops_leading_discourse_words_from_multiword_fragments() {
    let parsed = parse_markdown_document(
        "As Yō tries to pry open the skylight, sand falls through the crack.\n\n\
         “How’s Yoshiko doing in there?”\n",
    );

    let mentions = harvest_mention_candidates(
        "chapters/chapter-8.md",
        DocumentArchetype::Manuscript,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"As Yō"));
    assert!(!surfaces.contains(&"How Yoshiko"));
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
fn structured_field_harvesting_suppresses_emoji_bearing_fields() {
    let parsed = parse_markdown_document(
        "Role: Operations Lead✨\n\nAlias: kohaku🙂\n\nOutcome: the crew reaches the station.\n",
    );

    let fields = harvest_structured_field_candidates(
        "story planning/emoji-fields.txt",
        DocumentArchetype::StoryPlanning,
        &parsed,
    );

    let field_pairs: Vec<_> = fields
        .iter()
        .map(|field| (field.label.as_str(), field.value.as_str()))
        .collect();

    assert!(!field_pairs.contains(&("Role", "Operations Lead✨")));
    assert!(!field_pairs.contains(&("Alias", "kohaku🙂")));
    assert!(field_pairs.contains(&("Outcome", "the crew reaches the station")));
}

#[test]
fn loose_note_harvesting_suppresses_label_like_singletons() {
    let parsed = parse_markdown_document(
        "History with Yō: Former junior officer on several missions.\n\n\
         Professional identity\n\n\
         Dynamic with Prologue crew\n\n\
         Liella Personality: Bright, earnest, quietly stubborn\n\n\
         Relationship to Yō\n",
    );

    let mentions = harvest_mention_candidates(
        "notes/crew-summary.txt",
        DocumentArchetype::LooseNote,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"History"));
    assert!(!surfaces.contains(&"Professional"));
    assert!(!surfaces.contains(&"Dynamic"));
    assert!(!surfaces.contains(&"Bright"));
    assert!(!surfaces.contains(&"Relationship"));
    assert!(surfaces.contains(&"Yō"));
    assert!(surfaces.contains(&"Prologue"));
}

#[test]
fn loose_note_harvesting_suppresses_list_item_singleton_noise() {
    let parsed = parse_markdown_document(
        "- Calm and collected on the bridge\n\
         - Known for treating her crew with gentleness\n\
         - Drives morale by optimism rather than force\n\
         - Loved by her officers\n\
         - Moves like someone always ready to react\n\
         - Often first to detect shipboard anomalies\n\
         - Yō keeps the bridge steady\n\
         - Chisato’s athletic injury manager\n",
    );

    let mentions = harvest_mention_candidates(
        "notes/prologue-summary.txt",
        DocumentArchetype::LooseNote,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"Calm"));
    assert!(!surfaces.contains(&"Known"));
    assert!(!surfaces.contains(&"Drives"));
    assert!(!surfaces.contains(&"Loved"));
    assert!(!surfaces.contains(&"Moves"));
    assert!(!surfaces.contains(&"Often"));
    assert!(surfaces.contains(&"Yō"));
    assert!(surfaces.contains(&"Chisato"));
}

#[test]
fn loose_note_harvesting_suppresses_line_level_descriptor_and_field_labels() {
    let parsed = parse_markdown_document(
        "Bridge note\n\n\
         Childhood friend of Chisato\n\
         Liella Personality: Bright, earnest, quietly stubborn\n\
         History with Yō: Former junior officer on several missions\n\
         Chisato coordinates the bridge team\n\
         Yō reviews the route projection\n",
    );

    let mentions = harvest_mention_candidates(
        "notes/bridge-note.txt",
        DocumentArchetype::LooseNote,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"Childhood"));
    assert!(!surfaces.contains(&"Liella Personality"));
    assert!(!surfaces.contains(&"History"));
    assert!(surfaces.contains(&"Chisato"));
    assert!(surfaces.contains(&"Yō"));
}

#[test]
fn loose_note_harvesting_suppresses_generic_bullet_start_singletons() {
    let parsed = parse_markdown_document(
        "- Overprepares for disaster\n\
         - Accidentally comic relief\n\
         - Frequently mediates between chaos and procedure\n\
         - Perfect chaos partner for Keke\n\
         - Requested Prologue assignment specifically to support Yō\n",
    );

    let mentions = harvest_mention_candidates(
        "notes/prologue-bullets.txt",
        DocumentArchetype::LooseNote,
        &parsed,
    );

    let surfaces: Vec<_> = mentions.iter().map(|candidate| candidate.surface.as_str()).collect();

    assert!(!surfaces.contains(&"Overprepares"));
    assert!(!surfaces.contains(&"Accidentally"));
    assert!(!surfaces.contains(&"Frequently"));
    assert!(!surfaces.contains(&"Perfect"));
    assert!(surfaces.contains(&"Yō"));
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
