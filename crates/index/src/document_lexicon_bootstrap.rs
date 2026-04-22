use std::collections::{HashMap, HashSet};

use writing_assist_core::{
    BootstrappedLexiconEntry, DefinitionCandidate, DocumentArchetype, MentionCandidate,
    MentionCluster, MentionFeature, ParsedMarkdownDocument, SectionSummarySeed,
    StructuredFieldCandidate, TargetAnchor,
};

use crate::{
    cluster_document_mentions, compile_exact_phrase_lexicon_matcher, harvest_definition_candidates,
    harvest_exact_phrase_lexicon_mentions, harvest_mention_candidates,
    harvest_section_summary_seeds, harvest_structured_field_candidates,
    induce_bootstrapped_lexicon_entries,
};

#[derive(Debug, Clone)]
pub struct DocumentLexiconBootstrapPass {
    pub pass_index: usize,
    pub mention_count_before_exact_phrase_matcher: usize,
    pub exact_phrase_match_count: usize,
    pub mention_count_after_merge: usize,
    pub new_surfaces_added: usize,
    pub material_changes_detected: bool,
    pub cluster_count: usize,
    pub bootstrapped_entry_count: usize,
}

#[derive(Debug, Clone)]
pub struct IteratedDocumentLexiconBootstrap {
    pub fields: Vec<StructuredFieldCandidate>,
    pub definitions: Vec<DefinitionCandidate>,
    pub section_summary_seeds: Vec<SectionSummarySeed>,
    pub passes: Vec<DocumentLexiconBootstrapPass>,
    pub final_mentions: Vec<MentionCandidate>,
    pub final_clusters: Vec<MentionCluster>,
    pub final_entries: Vec<BootstrappedLexiconEntry>,
    pub converged: bool,
}

/// Run bounded seedless lexicon induction passes for one document.
///
/// This is the first convergence slice of Phase 3.7b. It keeps the loop
/// document-local and deterministic: first-pass mention harvesting builds the
/// bootstrapped lexicon, the compiled exact-phrase matcher contributes more
/// mention evidence, and the process repeats until the merge stabilizes or the
/// pass limit is reached.
pub fn iterate_document_lexicon_bootstrap(
    document_path: impl AsRef<str>,
    archetype: DocumentArchetype,
    parsed: &ParsedMarkdownDocument,
    max_passes: usize,
) -> IteratedDocumentLexiconBootstrap {
    let document_path = document_path.as_ref();
    let fields = harvest_structured_field_candidates(document_path, archetype.clone(), parsed);
    let definitions = harvest_definition_candidates(document_path, archetype.clone(), parsed);
    let section_summary_seeds =
        harvest_section_summary_seeds(document_path, archetype.clone(), parsed);

    let mut current_mentions = harvest_mention_candidates(document_path, archetype.clone(), parsed);
    let mut passes = Vec::new();
    let mut final_clusters = Vec::new();
    let mut final_entries = Vec::new();
    let mut converged = false;

    for pass_index in 1..=max_passes {
        let clusters = cluster_document_mentions(
            document_path,
            archetype.clone(),
            &current_mentions,
            &fields,
            &definitions,
            &section_summary_seeds,
        );
        let entries =
            induce_bootstrapped_lexicon_entries(document_path, &clusters, &fields, &definitions);
        let matcher = compile_exact_phrase_lexicon_matcher(&entries);
        let exact_phrase_mentions = harvest_exact_phrase_lexicon_mentions(
            document_path,
            archetype.clone(),
            parsed,
            &matcher,
        );
        let before_surfaces = normalized_surfaces(&current_mentions);
        let (merged_mentions, changed) =
            merge_mention_candidates(&current_mentions, &exact_phrase_mentions);
        let after_surfaces = normalized_surfaces(&merged_mentions);
        let new_surfaces_added = after_surfaces.difference(&before_surfaces).count();

        passes.push(DocumentLexiconBootstrapPass {
            pass_index,
            mention_count_before_exact_phrase_matcher: current_mentions.len(),
            exact_phrase_match_count: exact_phrase_mentions.len(),
            mention_count_after_merge: merged_mentions.len(),
            new_surfaces_added,
            material_changes_detected: changed,
            cluster_count: clusters.len(),
            bootstrapped_entry_count: entries.len(),
        });

        final_clusters = clusters;
        final_entries = entries;

        if !changed {
            converged = true;
            current_mentions = merged_mentions;
            break;
        }

        current_mentions = merged_mentions;
    }

    IteratedDocumentLexiconBootstrap {
        fields,
        definitions,
        section_summary_seeds,
        passes,
        final_mentions: current_mentions,
        final_clusters,
        final_entries,
        converged,
    }
}

fn normalized_surfaces(mentions: &[MentionCandidate]) -> HashSet<String> {
    mentions
        .iter()
        .map(|candidate| candidate.normalized_surface.clone())
        .collect()
}

fn merge_mention_candidates(
    primary: &[MentionCandidate],
    secondary: &[MentionCandidate],
) -> (Vec<MentionCandidate>, bool) {
    let mut merged = Vec::<MentionCandidate>::new();
    let mut index_by_normalized_surface = HashMap::<String, usize>::new();
    let mut changed = false;

    for candidate in primary.iter().chain(secondary.iter()) {
        if let Some(existing_index) = index_by_normalized_surface
            .get(&candidate.normalized_surface)
            .copied()
        {
            let existing = &mut merged[existing_index];
            changed |= merge_anchors(&mut existing.source.anchors, &candidate.source.anchors);
            changed |= merge_occurrences(existing, candidate);
            changed |= merge_features(
                &mut existing.aggregate_features,
                &candidate.aggregate_features,
            );
        } else {
            index_by_normalized_surface.insert(candidate.normalized_surface.clone(), merged.len());
            merged.push(candidate.clone());
            if merged.len() > primary.len() {
                changed = true;
            }
        }
    }

    (merged, changed)
}

fn merge_anchors(existing: &mut Vec<TargetAnchor>, incoming: &[TargetAnchor]) -> bool {
    let mut changed = false;

    for anchor in incoming {
        if !existing.contains(anchor) {
            existing.push(anchor.clone());
            changed = true;
        }
    }

    changed
}

fn merge_occurrences(existing: &mut MentionCandidate, incoming: &MentionCandidate) -> bool {
    let mut changed = false;

    for occurrence in &incoming.occurrences {
        if !existing.occurrences.contains(occurrence) {
            existing.occurrences.push(occurrence.clone());
            changed = true;
        }
    }

    changed
}

fn merge_features(existing: &mut Vec<MentionFeature>, incoming: &[MentionFeature]) -> bool {
    let mut changed = false;

    for feature in incoming {
        if !existing.contains(feature) {
            existing.push(feature.clone());
            changed = true;
        }
    }

    changed
}
