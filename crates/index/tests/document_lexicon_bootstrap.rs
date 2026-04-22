use writing_assist_core::DocumentArchetype;
use writing_assist_index::{iterate_document_lexicon_bootstrap, parse_markdown_document};

#[test]
fn document_iteration_converges_after_recovering_lowercase_mentions() {
    let parsed = parse_markdown_document(
        "# Radiant Firth\n\nRole: Scout vessel\n\nBy dawn, the radiant firth had vanished into weather.\n",
    );

    let result = iterate_document_lexicon_bootstrap(
        "story planning/fleet-notes.md",
        DocumentArchetype::DossierProfile,
        &parsed,
        3,
    );

    assert!(result.converged);
    assert_eq!(result.passes.len(), 2);
    assert!(
        result
            .final_mentions
            .iter()
            .any(|candidate| candidate.normalized_surface == "radiant firth")
    );
    assert!(result.passes[0].material_changes_detected);
    assert!(!result.passes[1].material_changes_detected);
}

#[test]
fn document_iteration_respects_the_max_pass_limit() {
    let parsed = parse_markdown_document(
        "# Radiant Firth\n\nRole: Scout vessel\n\nBy dawn, the radiant firth had vanished into weather.\n",
    );

    let result = iterate_document_lexicon_bootstrap(
        "story planning/fleet-notes.md",
        DocumentArchetype::DossierProfile,
        &parsed,
        1,
    );

    assert!(!result.converged);
    assert_eq!(result.passes.len(), 1);
    assert!(
        result
            .final_mentions
            .iter()
            .any(|candidate| candidate.normalized_surface == "radiant firth")
    );
}

#[test]
fn document_iteration_converges_immediately_when_no_new_lexicon_supported_mentions_exist() {
    let parsed = parse_markdown_document("the harbor was quiet before dawn.\n");

    let result = iterate_document_lexicon_bootstrap(
        "chapters/quiet-harbor.md",
        DocumentArchetype::Manuscript,
        &parsed,
        3,
    );

    assert!(result.converged);
    assert_eq!(result.passes.len(), 1);
    assert!(!result.passes[0].material_changes_detected);
    assert_eq!(result.passes[0].new_surfaces_added, 0);
}
