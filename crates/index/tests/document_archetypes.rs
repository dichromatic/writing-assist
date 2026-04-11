use std::fs;
use std::path::PathBuf;

use writing_assist_core::{DocumentArchetype, DocumentType};
use writing_assist_index::{classify_document_archetype, parse_markdown_document};

fn example_path(relative_path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(relative_path)
}

fn classify_fixture(path: &str, document_type: DocumentType) -> DocumentArchetype {
    let file_path = example_path(path);
    let text = fs::read_to_string(&file_path).expect("fixture should load");
    let parsed = parse_markdown_document(&text);

    classify_document_archetype(document_type, path, &text, &parsed)
}

#[test]
fn classifies_profile_and_planning_fixtures_by_structure() {
    assert_eq!(
        classify_fixture("story planning/estuary crew summaries.txt", DocumentType::Reference),
        DocumentArchetype::DossierProfile
    );
    assert_eq!(
        classify_fixture("story planning/prologue crew summaries.txt", DocumentType::Reference),
        DocumentArchetype::DossierProfile
    );
    assert_eq!(
        classify_fixture("story planning/recovery arcs.txt", DocumentType::Note),
        DocumentArchetype::StoryPlanning
    );
    assert_eq!(
        classify_fixture(
            "story planning/briefing for mission chapter 10-11.txt",
            DocumentType::Note
        ),
        DocumentArchetype::StoryPlanning
    );
}

#[test]
fn classifies_taxonomy_and_world_article_fixtures_differently() {
    assert_eq!(
        classify_fixture("world context/tau sectors.txt", DocumentType::Reference),
        DocumentArchetype::TaxonomyReference
    );
    assert_eq!(
        classify_fixture("world context/human history.txt", DocumentType::Reference),
        DocumentArchetype::ExpositoryWorldArticle
    );
    assert_eq!(
        classify_fixture(
            "world context/triumvirate and sectoring.txt",
            DocumentType::Reference
        ),
        DocumentArchetype::ExpositoryWorldArticle
    );
}

#[test]
fn classifies_generalized_structures_without_example_specific_keywords() {
    let dossier = "ALINA VOSS — OPERATIONS LEAD\n\nRole: Operations Lead\nBackground: Former station pilot.\nTraits: precise, reserved, loyal.\nRelationships\n- Mira: oldest friend\n- Tomas: frequent collaborator\n";
    let dossier_parsed = parse_markdown_document(dossier);
    assert_eq!(
        classify_document_archetype(
            DocumentType::Reference,
            "reference/crew-profiles.txt",
            dossier,
            &dossier_parsed
        ),
        DocumentArchetype::DossierProfile
    );

    let planning = "ACT II — Breach\n\nTone: tense, intimate\nGoal: isolate the team before the reveal.\n\n1. Arrival at the station\n- establish unease\n- delay the briefing\n\nOutcome: the group splits under pressure.\n";
    let planning_parsed = parse_markdown_document(planning);
    assert_eq!(
        classify_document_archetype(
            DocumentType::Note,
            "notes/act-2-outline.txt",
            planning,
            &planning_parsed
        ),
        DocumentArchetype::StoryPlanning
    );

    let taxonomy = "FIELD STATES\n\nState 1\n- stable resonance\n- low drift\n\nState 2\n- elevated drift\n- severe interference\n\nBoundaries\n- transitions create pressure and heat spikes\n";
    let taxonomy_parsed = parse_markdown_document(taxonomy);
    assert_eq!(
        classify_document_archetype(
            DocumentType::Reference,
            "reference/system-glossary.txt",
            taxonomy,
            &taxonomy_parsed
        ),
        DocumentArchetype::TaxonomyReference
    );

    let article = "### Origins\n\nThe coastal federation formed slowly across several centuries of migration, trade, and negotiated law. Its institutions were never designed in a single moment, but accreted through repeated settlement crises and long reform movements.\n\n### Expansion\n\nBy the late orbital era, the federation had become a stable interregional power whose influence depended more on logistics and civic trust than on conquest or coercion.\n";
    let article_parsed = parse_markdown_document(article);
    assert_eq!(
        classify_document_archetype(
            DocumentType::Reference,
            "reference/political-history.txt",
            article,
            &article_parsed
        ),
        DocumentArchetype::ExpositoryWorldArticle
    );
}

#[test]
fn manuscript_documents_remain_manuscripts_even_if_they_have_structure() {
    let manuscript = "# Chapter One\n\nCaptain Mara reaches the harbor.\n";
    let parsed = parse_markdown_document(manuscript);

    assert_eq!(
        classify_document_archetype(
            DocumentType::Manuscript,
            "chapters/chapter-1.md",
            manuscript,
            &parsed
        ),
        DocumentArchetype::Manuscript
    );
}

#[test]
fn ambiguous_note_like_documents_fall_back_to_loose_notes() {
    let note = "remember to revise the harbor scene later\nmaybe tie Dia into this somehow\n";
    let parsed = parse_markdown_document(note);

    assert_eq!(
        classify_document_archetype(DocumentType::Note, "notes/scratch.txt", note, &parsed),
        DocumentArchetype::LooseNote
    );
}
