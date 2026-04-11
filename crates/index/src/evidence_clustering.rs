use uuid::Uuid;
use writing_assist_core::{
    DefinitionCandidate, DocumentArchetype, MemorySourceReference, MentionCandidate,
    MentionCluster, MentionClusterLink, MentionClusterLinkKind, MentionFeature,
    MentionOccurrence, SectionSummarySeed, StructuredFieldCandidate, TargetAnchor,
};

/// Group same-document mention evidence into deterministic local clusters.
///
/// This stays intentionally conservative: it only merges exact normalized
/// surfaces and obvious title-to-bare-name variants, then links the resulting
/// cluster to other evidence records from the same document.
pub fn cluster_document_mentions(
    document_path: impl AsRef<str>,
    archetype: DocumentArchetype,
    mentions: &[MentionCandidate],
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
    summary_seeds: &[SectionSummarySeed],
) -> Vec<MentionCluster> {
    let document_path = document_path.as_ref();
    let mut groups = Vec::<MentionClusterGroup>::new();

    for mention in mentions {
        let cluster_key = cluster_key_for_mention(mention);

        if let Some(existing_group) = groups
            .iter_mut()
            .find(|group| should_merge_into_group(group, &cluster_key, mention))
        {
            existing_group.member_mentions.push(mention.clone());
            merge_unique_strings(
                &mut existing_group.cluster_keys,
                std::slice::from_ref(&cluster_key),
            );
        } else {
            groups.push(MentionClusterGroup {
                cluster_keys: vec![cluster_key],
                member_mentions: vec![mention.clone()],
            });
        }
    }

    groups
        .into_iter()
        .map(|group| build_cluster(
            document_path,
            archetype.clone(),
            &group,
            structured_fields,
            definitions,
            summary_seeds,
        ))
        .collect()
}

#[derive(Debug, Clone)]
struct MentionClusterGroup {
    cluster_keys: Vec<String>,
    member_mentions: Vec<MentionCandidate>,
}

fn build_cluster(
    document_path: &str,
    archetype: DocumentArchetype,
    group: &MentionClusterGroup,
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
    summary_seeds: &[SectionSummarySeed],
) -> MentionCluster {
    let display_surface = choose_display_surface(&group.member_mentions);
    let normalized_surface = cluster_key_for_surface(&display_surface);
    let source = combined_source(document_path, &group.member_mentions);
    let member_mention_ids = group.member_mentions.iter().map(|mention| mention.id).collect();
    let member_surfaces = unique_member_surfaces(&group.member_mentions);
    let occurrences = combined_occurrences(&group.member_mentions);
    let aggregate_features = combined_features(&group.member_mentions, occurrences.len());
    let linked_evidence = linked_evidence_for_cluster(
        group,
        structured_fields,
        definitions,
        summary_seeds,
    );

    MentionCluster {
        id: stable_hash_id(document_path, "mention_cluster", &normalized_surface, &display_surface),
        display_surface,
        normalized_surface,
        source,
        member_mention_ids,
        member_surfaces,
        occurrences,
        aggregate_features,
        linked_evidence,
        archetype,
    }
}

fn should_merge_into_group(
    group: &MentionClusterGroup,
    cluster_key: &str,
    mention: &MentionCandidate,
) -> bool {
    if group.cluster_keys.iter().any(|existing| existing == cluster_key) {
        return true;
    }

    group.member_mentions.iter().any(|existing| {
        let existing_titleless = titleless_surface(existing.surface.as_str());
        let candidate_titleless = titleless_surface(mention.surface.as_str());

        !existing_titleless.is_empty() && existing_titleless == candidate_titleless
            || !candidate_titleless.is_empty()
                && candidate_titleless == existing.normalized_surface
            || !existing_titleless.is_empty() && existing_titleless == mention.normalized_surface
    })
}

fn choose_display_surface(mentions: &[MentionCandidate]) -> String {
    mentions
        .iter()
        .max_by_key(|mention| {
            (
                mention
                    .aggregate_features
                    .contains(&MentionFeature::Titled),
                mention
                    .aggregate_features
                    .contains(&MentionFeature::MultiWord),
                mention.occurrences.len(),
                mention.surface.len(),
            )
        })
        .map(|mention| mention.surface.clone())
        .unwrap_or_default()
}

fn combined_source(document_path: &str, mentions: &[MentionCandidate]) -> MemorySourceReference {
    let mut anchors = Vec::new();
    let mut start_char = usize::MAX;
    let mut end_char = 0;

    for mention in mentions {
        merge_unique_anchors(&mut anchors, &mention.source.anchors);
        start_char = start_char.min(mention.source.start_char);
        end_char = end_char.max(mention.source.end_char);
    }

    MemorySourceReference::new(
        document_path,
        anchors,
        if start_char == usize::MAX { 0 } else { start_char },
        end_char,
    )
}

fn unique_member_surfaces(mentions: &[MentionCandidate]) -> Vec<String> {
    let mut surfaces = Vec::new();

    for mention in mentions {
        if !surfaces.contains(&mention.surface) {
            surfaces.push(mention.surface.clone());
        }
    }

    surfaces
}

fn combined_occurrences(mentions: &[MentionCandidate]) -> Vec<MentionOccurrence> {
    let mut occurrences = Vec::new();

    for mention in mentions {
        for occurrence in &mention.occurrences {
            if !occurrences.contains(occurrence) {
                occurrences.push(occurrence.clone());
            }
        }
    }

    occurrences
}

fn combined_features(
    mentions: &[MentionCandidate],
    occurrence_count: usize,
) -> Vec<MentionFeature> {
    let mut features = Vec::new();

    for mention in mentions {
        for feature in &mention.aggregate_features {
            if !features.contains(feature) {
                features.push(feature.clone());
            }
        }
    }

    if occurrence_count > 1 && !features.contains(&MentionFeature::Repeated) {
        features.push(MentionFeature::Repeated);
    }

    features
}

fn linked_evidence_for_cluster(
    group: &MentionClusterGroup,
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
    summary_seeds: &[SectionSummarySeed],
) -> Vec<MentionClusterLink> {
    let mut links = Vec::new();
    let member_surfaces = group
        .member_mentions
        .iter()
        .map(|mention| mention.surface.as_str())
        .collect::<Vec<_>>();
    let normalized_surfaces = group
        .member_mentions
        .iter()
        .map(|mention| mention.normalized_surface.as_str())
        .collect::<Vec<_>>();
    let section_anchors = group
        .member_mentions
        .iter()
        .flat_map(|mention| mention.occurrences.iter())
        .filter_map(|occurrence| occurrence.section_anchor.clone())
        .collect::<Vec<_>>();

    for field in structured_fields {
        let heading_text = field
            .contexts
            .iter()
            .filter_map(|context| context.heading.as_deref())
            .collect::<Vec<_>>()
            .join(" ");
        let haystacks = [heading_text, field.value.clone(), field.label.clone()];

        if haystacks.iter().any(|haystack| {
            surface_matches_haystack(haystack, &member_surfaces, &normalized_surfaces)
        }) {
            links.push(MentionClusterLink {
                kind: MentionClusterLinkKind::StructuredField,
                evidence_id: field.id,
                summary: format!("{}: {}", field.label, field.value),
            });
        }
    }

    for definition in definitions {
        let haystacks = [definition.term.clone(), definition.definition.clone()];

        if haystacks.iter().any(|haystack| {
            surface_matches_haystack(haystack, &member_surfaces, &normalized_surfaces)
        }) {
            links.push(MentionClusterLink {
                kind: MentionClusterLinkKind::Definition,
                evidence_id: definition.id,
                summary: format!("{} => {}", definition.term, definition.definition),
            });
        }
    }

    for seed in summary_seeds {
        let seed_sections = seed
            .contexts
            .iter()
            .filter_map(|context| context.section_anchor.clone())
            .collect::<Vec<_>>();

        if seed_sections
            .iter()
            .any(|seed_section| section_anchors.contains(seed_section))
        {
            links.push(MentionClusterLink {
                kind: MentionClusterLinkKind::SectionSummarySeed,
                evidence_id: seed.id,
                summary: seed.scope.clone(),
            });
        }
    }

    dedupe_links(&mut links);
    links
}

fn surface_matches_haystack(
    haystack: &str,
    member_surfaces: &[&str],
    normalized_surfaces: &[&str],
) -> bool {
    let normalized_haystack = normalize_for_match(haystack);

    member_surfaces.iter().any(|surface| {
        let normalized_surface = normalize_for_match(surface);

        !normalized_surface.is_empty() && normalized_haystack.contains(&normalized_surface)
    }) || normalized_surfaces.iter().any(|surface| {
        !surface.is_empty() && normalized_haystack.contains(surface)
    })
}

fn dedupe_links(links: &mut Vec<MentionClusterLink>) {
    let mut deduped = Vec::new();

    for link in links.drain(..) {
        if deduped.iter().any(|existing: &MentionClusterLink| {
            existing.kind == link.kind && existing.evidence_id == link.evidence_id
        }) {
            continue;
        }

        deduped.push(link);
    }

    *links = deduped;
}

fn cluster_key_for_mention(mention: &MentionCandidate) -> String {
    let titleless = titleless_surface(mention.surface.as_str());

    if !titleless.is_empty() {
        titleless
    } else {
        mention.normalized_surface.clone()
    }
}

fn cluster_key_for_surface(surface: &str) -> String {
    let titleless = titleless_surface(surface);

    if !titleless.is_empty() {
        titleless
    } else {
        normalize_for_match(surface)
    }
}

fn titleless_surface(surface: &str) -> String {
    let words = surface.split_whitespace().collect::<Vec<_>>();

    if words.len() < 2 || !is_title_prefix(words[0]) {
        return String::new();
    }

    words[1..]
        .iter()
        .map(|word| normalize_for_match(word))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_for_match(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|character| {
            character.is_alphanumeric() || character.is_whitespace() || matches!(character, '-' | '\'')
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_title_prefix(token: &str) -> bool {
    matches!(
        token,
        "Captain"
            | "Admiral"
            | "Commander"
            | "Dr"
            | "Doctor"
            | "Professor"
            | "Master"
            | "Mrs"
            | "Miss"
            | "Mr"
            | "Ms"
            | "Archmage"
            | "Pioneer"
            | "General"
            | "Elder"
    )
}

fn merge_unique_strings(existing: &mut Vec<String>, incoming: &[String]) {
    for value in incoming {
        if !existing.contains(value) {
            existing.push(value.clone());
        }
    }
}

fn merge_unique_anchors(existing: &mut Vec<TargetAnchor>, incoming: &[TargetAnchor]) {
    for anchor in incoming {
        if !existing.contains(anchor) {
            existing.push(anchor.clone());
        }
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
