use std::collections::HashSet;

use uuid::Uuid;
use writing_assist_core::{
    BootstrappedLexiconEntry, DefinitionCandidate, EvidenceSuppressionReason, MentionCluster,
    MentionClusterLinkKind, MentionFeature, PromotedEvidenceBundle, PromotedEvidenceCandidate,
    PromotedEvidenceKind, PromotionReason, StructuredFieldCandidate, SuppressedEvidenceCandidate,
};

/// Promote raw deterministic evidence into compact retrieval/semantic input.
///
/// Phase 3.7d deliberately stays below semantic interpretation: relationship-
/// shaped fields can be retained for review, but this module does not emit
/// final relationship records or infer complex indexes.
pub fn promote_evidence_for_context(
    clusters: &[MentionCluster],
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
    lexicon_entries: &[BootstrappedLexiconEntry],
) -> PromotedEvidenceBundle {
    let definition_ids = definitions
        .iter()
        .map(|definition| definition.id)
        .collect::<HashSet<_>>();
    let field_ids = structured_fields
        .iter()
        .map(|field| field.id)
        .collect::<HashSet<_>>();
    let lexicon_surfaces = lexicon_entries
        .iter()
        .map(|entry| entry.normalized_surface.as_str())
        .collect::<HashSet<_>>();

    let mut promoted = Vec::new();
    let mut review_only = relationship_fields_review_only(structured_fields);
    let mut suppressed = Vec::new();

    for cluster in clusters {
        let support =
            cluster_promotion_support(cluster, &definition_ids, &field_ids, &lexicon_surfaces);

        if support.unresolved_abbreviation {
            suppressed.push(suppressed_cluster(
                cluster,
                EvidenceSuppressionReason::UnresolvedAbbreviation,
            ));
            continue;
        }

        if support.weak_singleton {
            suppressed.push(suppressed_cluster(
                cluster,
                EvidenceSuppressionReason::WeakSingleton,
            ));
            continue;
        }

        let Some(candidate) = promoted_cluster(cluster, &support) else {
            continue;
        };

        if candidate.kind == PromotedEvidenceKind::ReviewOnly {
            review_only.push(candidate);
        } else {
            promoted.push(candidate);
        }
    }

    PromotedEvidenceBundle {
        promoted,
        review_only,
        suppressed,
    }
}

#[derive(Debug, Clone)]
struct ClusterPromotionSupport {
    repeated: bool,
    multi_word: bool,
    titled: bool,
    definition_backed: bool,
    field_backed: bool,
    lexicon_backed: bool,
    weak_singleton: bool,
    unresolved_abbreviation: bool,
}

fn cluster_promotion_support(
    cluster: &MentionCluster,
    definition_ids: &HashSet<Uuid>,
    field_ids: &HashSet<Uuid>,
    lexicon_surfaces: &HashSet<&str>,
) -> ClusterPromotionSupport {
    let repeated = cluster
        .aggregate_features
        .contains(&MentionFeature::Repeated)
        || cluster.occurrences.len() > 1;
    let multi_word = cluster
        .aggregate_features
        .contains(&MentionFeature::MultiWord)
        || cluster.display_surface.split_whitespace().count() > 1;
    let titled = cluster.aggregate_features.contains(&MentionFeature::Titled);
    let definition_backed = cluster.linked_evidence.iter().any(|link| {
        link.kind == MentionClusterLinkKind::Definition
            && definition_ids.contains(&link.evidence_id)
    });
    let field_backed = cluster.linked_evidence.iter().any(|link| {
        link.kind == MentionClusterLinkKind::StructuredField
            && field_ids.contains(&link.evidence_id)
    });
    let lexicon_backed = lexicon_surfaces.contains(cluster.normalized_surface.as_str());
    let strong_support =
        repeated || multi_word || titled || definition_backed || field_backed || lexicon_backed;
    let abbreviation_has_grounding = definition_backed || field_backed || lexicon_backed;

    ClusterPromotionSupport {
        repeated,
        multi_word,
        titled,
        definition_backed,
        field_backed,
        lexicon_backed,
        weak_singleton: !strong_support,
        unresolved_abbreviation: is_unresolved_abbreviation(cluster) && !abbreviation_has_grounding,
    }
}

fn promoted_cluster(
    cluster: &MentionCluster,
    support: &ClusterPromotionSupport,
) -> Option<PromotedEvidenceCandidate> {
    let mut reasons = promotion_reasons(support);
    if reasons.is_empty() {
        return None;
    }

    let kind = if support.definition_backed {
        PromotedEvidenceKind::Terminology
    } else if support.field_backed {
        PromotedEvidenceKind::FieldBackedContext
    } else {
        PromotedEvidenceKind::EntityLike
    };

    // Keep reason order stable for snapshots, logs, and future prompt packets.
    reasons.sort_by_key(reason_sort_key);

    Some(PromotedEvidenceCandidate {
        id: stable_hash_id(
            &cluster.source.document_path,
            "promoted_evidence",
            &cluster.normalized_surface,
            &cluster.display_surface,
        ),
        display_surface: cluster.display_surface.clone(),
        normalized_surface: cluster.normalized_surface.clone(),
        kind,
        source: cluster.source.clone(),
        archetype: cluster.archetype.clone(),
        evidence_ids: cluster_evidence_ids(cluster),
        reasons,
        snippets: representative_snippets(cluster),
    })
}

fn relationship_fields_review_only(
    structured_fields: &[StructuredFieldCandidate],
) -> Vec<PromotedEvidenceCandidate> {
    structured_fields
        .iter()
        .filter(|field| is_relationship_like_label(&field.label))
        .map(|field| {
            let display_surface = format!("{}: {}", field.label, field.value);
            let normalized_surface = normalize_surface(&display_surface);

            PromotedEvidenceCandidate {
                id: stable_hash_id(
                    &field.source.document_path,
                    "relationship_field_review_only",
                    &normalized_surface,
                    &display_surface,
                ),
                display_surface,
                normalized_surface,
                kind: PromotedEvidenceKind::FieldBackedContext,
                source: field.source.clone(),
                archetype: field.archetype.clone(),
                evidence_ids: vec![field.id],
                reasons: vec![PromotionReason::FieldBacked],
                snippets: field
                    .contexts
                    .iter()
                    .map(|context| context.excerpt.clone())
                    .filter(|snippet| !snippet.is_empty())
                    .take(2)
                    .collect(),
            }
        })
        .collect()
}

fn suppressed_cluster(
    cluster: &MentionCluster,
    reason: EvidenceSuppressionReason,
) -> SuppressedEvidenceCandidate {
    SuppressedEvidenceCandidate {
        evidence_id: cluster.id,
        display_surface: cluster.display_surface.clone(),
        normalized_surface: cluster.normalized_surface.clone(),
        source: cluster.source.clone(),
        archetype: cluster.archetype.clone(),
        reason,
    }
}

fn promotion_reasons(support: &ClusterPromotionSupport) -> Vec<PromotionReason> {
    let mut reasons = Vec::new();

    if support.repeated {
        reasons.push(PromotionReason::RepeatedMention);
    }
    if support.multi_word {
        reasons.push(PromotionReason::MultiWordMention);
    }
    if support.titled {
        reasons.push(PromotionReason::TitledMention);
    }
    if support.definition_backed {
        reasons.push(PromotionReason::DefinitionBacked);
    }
    if support.field_backed {
        reasons.push(PromotionReason::FieldBacked);
    }
    if support.lexicon_backed {
        reasons.push(PromotionReason::LexiconBacked);
    }

    reasons
}

fn cluster_evidence_ids(cluster: &MentionCluster) -> Vec<Uuid> {
    let mut ids = vec![cluster.id];

    for link in &cluster.linked_evidence {
        if !ids.contains(&link.evidence_id) {
            ids.push(link.evidence_id);
        }
    }

    ids
}

fn representative_snippets(cluster: &MentionCluster) -> Vec<String> {
    let mut snippets = Vec::new();

    for occurrence in &cluster.occurrences {
        if !occurrence.snippet.is_empty() && !snippets.contains(&occurrence.snippet) {
            snippets.push(occurrence.snippet.clone());
        }

        if snippets.len() >= 2 {
            break;
        }
    }

    snippets
}

fn is_unresolved_abbreviation(cluster: &MentionCluster) -> bool {
    let compact = cluster.display_surface.trim();

    compact.len() >= 2
        && compact.len() <= 8
        && compact
            .chars()
            .all(|character| character.is_ascii_uppercase() || character.is_ascii_digit())
        && compact
            .chars()
            .any(|character| character.is_ascii_uppercase())
}

fn is_relationship_like_label(label: &str) -> bool {
    matches!(
        label.to_ascii_lowercase().as_str(),
        "relationship" | "relationships" | "mentor" | "mentors" | "ally" | "allies"
    )
}

fn normalize_surface(surface: &str) -> String {
    surface
        .to_lowercase()
        .chars()
        .filter(|character| {
            character.is_alphanumeric()
                || character.is_whitespace()
                || matches!(character, '-' | '\'')
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn reason_sort_key(reason: &PromotionReason) -> usize {
    match reason {
        PromotionReason::RepeatedMention => 0,
        PromotionReason::MultiWordMention => 1,
        PromotionReason::TitledMention => 2,
        PromotionReason::DefinitionBacked => 3,
        PromotionReason::FieldBacked => 4,
        PromotionReason::LexiconBacked => 5,
    }
}

fn stable_hash_id(document_path: &str, kind: &str, left: &str, right: &str) -> Uuid {
    let mut hash = 0xcbf29ce484222325_u128;

    for byte in document_path
        .bytes()
        .chain([0])
        .chain(kind.bytes())
        .chain([0])
        .chain(left.bytes())
        .chain([0])
        .chain(right.bytes())
    {
        hash ^= byte as u128;
        hash = hash.wrapping_mul(0x00000100000001b3);
    }

    Uuid::from_u128(hash)
}
