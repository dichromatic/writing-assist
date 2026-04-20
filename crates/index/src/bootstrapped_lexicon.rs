use uuid::Uuid;
use writing_assist_core::{
    DefinitionCandidate, DocumentArchetype, MentionCluster, MentionClusterLinkKind,
    MentionFeature, BootstrappedLexiconEntry, BootstrappedLexiconEntryKind,
    LexiconSupportRecord, LexiconSupportRecordKind, LexiconBootstrapRule,
    StructuredFieldCandidate,
};

/// Build deterministic bootstrapped lexicon entries from same-document evidence.
///
/// This is the first seedless lexicon slice for Phase 3.7b. It intentionally
/// stops at document-local induction so later project-level passes can merge
/// and rerank entries without coupling this module to cross-document identity.
pub fn induce_bootstrapped_lexicon_entries(
    document_path: impl AsRef<str>,
    mention_clusters: &[MentionCluster],
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
) -> Vec<BootstrappedLexiconEntry> {
    let document_path = document_path.as_ref();

    mention_clusters
        .iter()
        .map(|cluster| {
            let evidence = evidence_for_cluster(cluster, structured_fields, definitions);
            let rule_sources = rule_sources_for_cluster(cluster, structured_fields, definitions);

            BootstrappedLexiconEntry {
                id: stable_hash_id(
                    document_path,
                    "bootstrapped_lexicon_entry",
                    &cluster.normalized_surface,
                    &cluster.display_surface,
                ),
                canonical_surface: cluster.display_surface.clone(),
                normalized_surface: cluster.normalized_surface.clone(),
                kind: entry_kind_for_cluster(cluster, structured_fields, definitions),
                source: cluster.source.clone(),
                occurrence_count: cluster.occurrences.len(),
                archetypes_seen: vec![cluster.archetype.clone()],
                rule_sources,
                evidence,
            }
        })
        .collect()
}

fn entry_kind_for_cluster(
    cluster: &MentionCluster,
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
) -> BootstrappedLexiconEntryKind {
    if has_definition_support(cluster, definitions) {
        return BootstrappedLexiconEntryKind::Terminology;
    }

    if cluster
        .aggregate_features
        .contains(&MentionFeature::Titled)
        || has_alias_field_support(cluster, structured_fields)
        || has_participant_field_support(cluster, structured_fields)
    {
        return BootstrappedLexiconEntryKind::Character;
    }

    match cluster.archetype {
        DocumentArchetype::TaxonomyReference | DocumentArchetype::ExpositoryWorldArticle => {
            BootstrappedLexiconEntryKind::Terminology
        }
        _ => BootstrappedLexiconEntryKind::Unresolved,
    }
}

fn rule_sources_for_cluster(
    cluster: &MentionCluster,
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
) -> Vec<LexiconBootstrapRule> {
    let mut rules = Vec::new();

    if cluster.occurrences.len() > 1 {
        rules.push(LexiconBootstrapRule::RepeatedMention);
    }
    if cluster
        .aggregate_features
        .contains(&MentionFeature::Titled)
    {
        rules.push(LexiconBootstrapRule::TitledMention);
    }
    if has_alias_field_support(cluster, structured_fields) {
        rules.push(LexiconBootstrapRule::AliasField);
    }
    if has_participant_field_support(cluster, structured_fields) {
        rules.push(LexiconBootstrapRule::ParticipantField);
    }
    if has_linked_structured_field(cluster) {
        rules.push(LexiconBootstrapRule::LinkedStructuredField);
    }
    if has_definition_support(cluster, definitions) {
        rules.push(LexiconBootstrapRule::DefinitionTerm);
        rules.push(LexiconBootstrapRule::LinkedDefinition);
    }

    rules
}

fn evidence_for_cluster(
    cluster: &MentionCluster,
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
) -> Vec<LexiconSupportRecord> {
    let mut evidence = vec![LexiconSupportRecord {
        evidence_id: cluster.id,
        kind: LexiconSupportRecordKind::MentionCluster,
        summary: cluster.member_surfaces.join(" / "),
    }];

    for link in &cluster.linked_evidence {
        match link.kind {
            MentionClusterLinkKind::StructuredField => {
                if let Some(field) = structured_fields.iter().find(|field| field.id == link.evidence_id)
                {
                    evidence.push(LexiconSupportRecord {
                        evidence_id: field.id,
                        kind: LexiconSupportRecordKind::StructuredField,
                        summary: format!("{}: {}", field.label, field.value),
                    });
                }
            }
            MentionClusterLinkKind::Definition => {
                if let Some(definition) =
                    definitions.iter().find(|definition| definition.id == link.evidence_id)
                {
                    evidence.push(LexiconSupportRecord {
                        evidence_id: definition.id,
                        kind: LexiconSupportRecordKind::Definition,
                        summary: format!("{} => {}", definition.term, definition.definition),
                    });
                }
            }
            MentionClusterLinkKind::SectionSummarySeed => {}
        }
    }

    evidence
}

fn has_alias_field_support(
    cluster: &MentionCluster,
    structured_fields: &[StructuredFieldCandidate],
) -> bool {
    structured_fields.iter().any(|field| {
        cluster
            .linked_evidence
            .iter()
            .any(|link| link.evidence_id == field.id)
            && matches!(
                field.label.to_ascii_lowercase().as_str(),
                "alias" | "aliases" | "nickname" | "callsign"
            )
    })
}

fn has_participant_field_support(
    cluster: &MentionCluster,
    structured_fields: &[StructuredFieldCandidate],
) -> bool {
    structured_fields.iter().any(|field| {
        cluster
            .linked_evidence
            .iter()
            .any(|link| link.evidence_id == field.id)
            && matches!(
                field.label.to_ascii_lowercase().as_str(),
                "participant"
                    | "participants"
                    | "focus"
                    | "target"
                    | "character"
                    | "characters"
                    | "crew"
                    | "speaker"
                    | "speakers"
            )
    })
}

fn has_definition_support(cluster: &MentionCluster, definitions: &[DefinitionCandidate]) -> bool {
    definitions.iter().any(|definition| {
        cluster
            .linked_evidence
            .iter()
            .any(|link| link.evidence_id == definition.id)
    })
}

fn has_linked_structured_field(cluster: &MentionCluster) -> bool {
    cluster
        .linked_evidence
        .iter()
        .any(|link| link.kind == MentionClusterLinkKind::StructuredField)
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
