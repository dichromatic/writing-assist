use writing_assist_core::{
    DocumentArchetype, EntityCandidate, MemoryReviewState, MemoryStalenessState,
};

use crate::evidence_harvesting::harvest_mention_candidates;

/// Preserve the Phase 3.2 entity spike as a narrow promotion layer on top of
/// mention evidence.
///
/// The product direction is shifting toward evidence harvesting first and
/// semantic consolidation later. This function now keeps the old entity API
/// alive by promoting only the mention records that meet the original
/// "candidate entity" threshold.
pub fn extract_entity_candidates(
    document_path: impl AsRef<str>,
    parsed: &writing_assist_core::ParsedMarkdownDocument,
) -> Vec<EntityCandidate> {
    harvest_mention_candidates(document_path.as_ref(), DocumentArchetype::Manuscript, parsed)
        .into_iter()
        .filter(|candidate| {
            candidate.occurrences.len() > 1
                || candidate
                    .aggregate_features
                    .iter()
                    .any(|feature| {
                        matches!(
                            feature,
                            writing_assist_core::MentionFeature::MultiWord
                                | writing_assist_core::MentionFeature::Titled
                        )
                    })
        })
        .map(|candidate| {
            EntityCandidate::new(
                candidate.id,
                candidate.surface,
                candidate.source,
                MemoryReviewState::Pending,
                MemoryStalenessState::Current,
            )
        })
        .collect()
}
