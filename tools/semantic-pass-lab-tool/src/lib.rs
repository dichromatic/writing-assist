mod nim_client;

use serde::Serialize;
use serde_json::{json, Value};
use writing_assist_core::{
    DefinitionCandidate, DocumentArchetype, DocumentType, MentionCluster, MentionClusterLinkKind,
    MentionFeature, SectionSummarySeed, StructuredFieldCandidate,
};

pub use nim_client::{build_nim_request, execute_semantic_pass, NimLabConfig, SemanticPassLabResponse};

const MAX_PROMPT_LINKED_EVIDENCE_PER_CLUSTER: usize = 5;
const MAX_PROMPT_SNIPPETS_PER_CLUSTER: usize = 2;
const MAX_PROMPT_MANUSCRIPT_FIELDS: usize = 12;
const MAX_PROMPT_REFERENCE_FIELDS: usize = 24;
const MAX_PROMPT_NOTE_FIELDS: usize = 30;
const MAX_PROMPT_DEFINITIONS: usize = 20;
const MAX_PROMPT_SUMMARY_SEEDS: usize = 6;
const MAX_PROMPT_TEXT_CHARS: usize = 220;

/// The lab compares a broader prompt bundle against a narrower curated bundle.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticBundleShape {
    HighRecall,
    Curated,
}

/// The experiment now uses document-type-specific semantic tasks instead of one
/// generic consolidation prompt. This makes the prompt tighter and keeps
/// manuscript, planning, and reference behaviors distinct.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticPassTaskKind {
    ManuscriptEntityConsolidation,
    ReferenceKnowledgeConsolidation,
    NotePlanningConsolidation,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SemanticPassLabPacket {
    pub experiment_name: String,
    pub bundle_shape: SemanticBundleShape,
    pub task_kind: SemanticPassTaskKind,
    pub document_path: String,
    pub document_type: DocumentType,
    pub archetype: DocumentArchetype,
    pub prompt: String,
    pub output_contract: String,
    pub clusters: Vec<LabClusterRecord>,
    pub structured_fields: Vec<LabFieldRecord>,
    pub definitions: Vec<LabDefinitionRecord>,
    pub section_summary_seeds: Vec<LabSummarySeedRecord>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LabClusterRecord {
    pub id: String,
    pub display_surface: String,
    pub member_surfaces: Vec<String>,
    pub occurrence_count: usize,
    pub aggregate_features: Vec<MentionFeature>,
    pub linked_evidence_summaries: Vec<String>,
    pub representative_snippets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LabFieldRecord {
    pub id: String,
    pub label: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LabDefinitionRecord {
    pub id: String,
    pub term: String,
    pub definition: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LabSummarySeedRecord {
    pub id: String,
    pub scope: String,
    pub text: String,
}

pub fn build_semantic_pass_lab_packet(
    experiment_name: impl Into<String>,
    document_path: impl Into<String>,
    document_type: DocumentType,
    archetype: DocumentArchetype,
    bundle_shape: SemanticBundleShape,
    clusters: &[MentionCluster],
    structured_fields: &[StructuredFieldCandidate],
    definitions: &[DefinitionCandidate],
    section_summary_seeds: &[SectionSummarySeed],
) -> SemanticPassLabPacket {
    let experiment_name = experiment_name.into();
    let document_path = document_path.into();
    let task_kind = semantic_pass_task_kind(document_type.clone(), archetype.clone());
    let selected_clusters = select_clusters(clusters, bundle_shape, task_kind);
    let selected_field_ids = select_structured_field_ids(
        structured_fields,
        &selected_clusters,
        bundle_shape,
    );
    let selected_definition_ids = select_definition_ids(
        definitions,
        &selected_clusters,
        bundle_shape,
    );
    let selected_summary_seed_ids = select_summary_seed_ids(
        section_summary_seeds,
        &selected_clusters,
        bundle_shape,
    );

    let cluster_records = selected_clusters
        .iter()
        .map(|cluster| LabClusterRecord {
            id: cluster.id.to_string(),
            display_surface: cluster.display_surface.clone(),
            member_surfaces: cluster.member_surfaces.clone(),
            occurrence_count: cluster.occurrences.len(),
            aggregate_features: cluster.aggregate_features.clone(),
            linked_evidence_summaries: cluster
                .linked_evidence
                .iter()
                .map(|link| format!("{:?}: {}", link.kind, link.summary))
                .collect(),
            representative_snippets: representative_snippets(cluster),
        })
        .collect::<Vec<_>>();

    let field_records = structured_fields
        .iter()
        .filter(|field| selected_field_ids.contains(&field.id.to_string()))
        .map(|field| LabFieldRecord {
            id: field.id.to_string(),
            label: field.label.clone(),
            value: field.value.clone(),
        })
        .collect::<Vec<_>>();

    let definition_records = definitions
        .iter()
        .filter(|definition| selected_definition_ids.contains(&definition.id.to_string()))
        .map(|definition| LabDefinitionRecord {
            id: definition.id.to_string(),
            term: definition.term.clone(),
            definition: definition.definition.clone(),
        })
        .collect::<Vec<_>>();

    let summary_seed_records = section_summary_seeds
        .iter()
        .filter(|seed| selected_summary_seed_ids.contains(&seed.id.to_string()))
        .map(|seed| LabSummarySeedRecord {
            id: seed.id.to_string(),
            scope: seed.scope.clone(),
            text: seed.text.clone(),
        })
        .collect::<Vec<_>>();

    let output_contract = semantic_output_contract(task_kind);
    let prompt = build_prompt(
        &document_path,
        document_type.clone(),
        archetype.clone(),
        bundle_shape,
        task_kind,
        &cluster_records,
        &field_records,
        &definition_records,
        &summary_seed_records,
        &output_contract,
    );

    SemanticPassLabPacket {
        experiment_name,
        bundle_shape,
        task_kind,
        document_path,
        document_type,
        archetype,
        prompt,
        output_contract,
        clusters: cluster_records,
        structured_fields: field_records,
        definitions: definition_records,
        section_summary_seeds: summary_seed_records,
    }
}

fn select_clusters(
    clusters: &[MentionCluster],
    bundle_shape: SemanticBundleShape,
    task_kind: SemanticPassTaskKind,
) -> Vec<MentionCluster> {
    let mut ranked = clusters.to_vec();
    ranked.sort_by_key(|cluster| {
        (
            cluster_priority_score(cluster),
            cluster.occurrences.len(),
            cluster.display_surface.len(),
        )
    });
    ranked.reverse();

    match bundle_shape {
        SemanticBundleShape::HighRecall => ranked.into_iter().take(18).collect(),
        SemanticBundleShape::Curated => ranked
            .into_iter()
            .filter(|cluster| is_actionable_curated_surface(cluster, task_kind))
            .filter(is_strong_semantic_cluster)
            .take(10)
            .collect(),
    }
}

fn select_structured_field_ids(
    structured_fields: &[StructuredFieldCandidate],
    selected_clusters: &[MentionCluster],
    bundle_shape: SemanticBundleShape,
) -> Vec<String> {
    let linked_ids = selected_clusters
        .iter()
        .flat_map(|cluster| cluster.linked_evidence.iter())
        .filter(|link| link.kind == MentionClusterLinkKind::StructuredField)
        .map(|link| link.evidence_id.to_string())
        .collect::<Vec<_>>();

    let mut ids = structured_fields
        .iter()
        .filter(|field| linked_ids.contains(&field.id.to_string()))
        .map(|field| field.id.to_string())
        .collect::<Vec<_>>();

    if bundle_shape == SemanticBundleShape::HighRecall {
        for field in structured_fields.iter().take(8) {
            if !ids.contains(&field.id.to_string()) {
                ids.push(field.id.to_string());
            }
        }
    }

    ids
}

fn select_definition_ids(
    definitions: &[DefinitionCandidate],
    selected_clusters: &[MentionCluster],
    bundle_shape: SemanticBundleShape,
) -> Vec<String> {
    let linked_ids = selected_clusters
        .iter()
        .flat_map(|cluster| cluster.linked_evidence.iter())
        .filter(|link| link.kind == MentionClusterLinkKind::Definition)
        .map(|link| link.evidence_id.to_string())
        .collect::<Vec<_>>();

    let mut ids = definitions
        .iter()
        .filter(|definition| linked_ids.contains(&definition.id.to_string()))
        .map(|definition| definition.id.to_string())
        .collect::<Vec<_>>();

    if bundle_shape == SemanticBundleShape::HighRecall {
        for definition in definitions.iter().take(8) {
            if !ids.contains(&definition.id.to_string()) {
                ids.push(definition.id.to_string());
            }
        }
    }

    ids
}

fn select_summary_seed_ids(
    section_summary_seeds: &[SectionSummarySeed],
    selected_clusters: &[MentionCluster],
    bundle_shape: SemanticBundleShape,
) -> Vec<String> {
    let linked_ids = selected_clusters
        .iter()
        .flat_map(|cluster| cluster.linked_evidence.iter())
        .filter(|link| link.kind == MentionClusterLinkKind::SectionSummarySeed)
        .map(|link| link.evidence_id.to_string())
        .collect::<Vec<_>>();

    let mut ids = section_summary_seeds
        .iter()
        .filter(|seed| linked_ids.contains(&seed.id.to_string()))
        .map(|seed| seed.id.to_string())
        .collect::<Vec<_>>();

    if bundle_shape == SemanticBundleShape::HighRecall {
        for seed in section_summary_seeds.iter().take(6) {
            if !ids.contains(&seed.id.to_string()) {
                ids.push(seed.id.to_string());
            }
        }
    }

    ids
}

fn cluster_priority_score(cluster: &MentionCluster) -> usize {
    let mut score = cluster.occurrences.len() * 2 + cluster.linked_evidence.len() * 3;

    if cluster.aggregate_features.contains(&MentionFeature::Titled) {
        score += 4;
    }
    if cluster.aggregate_features.contains(&MentionFeature::MultiWord) {
        score += 3;
    }
    if cluster.aggregate_features.contains(&MentionFeature::Repeated) {
        score += 2;
    }
    if cluster.aggregate_features.contains(&MentionFeature::HeadingMentioned) {
        score += 1;
    }

    score
}

fn is_strong_semantic_cluster(cluster: &MentionCluster) -> bool {
    cluster.aggregate_features.contains(&MentionFeature::Repeated)
        || cluster.aggregate_features.contains(&MentionFeature::MultiWord)
        || cluster.aggregate_features.contains(&MentionFeature::Titled)
        || !cluster.linked_evidence.is_empty()
}

fn is_actionable_curated_surface(
    cluster: &MentionCluster,
    task_kind: SemanticPassTaskKind,
) -> bool {
    let surface = cluster.display_surface.trim();

    if is_single_ascii_uppercase(surface) {
        return false;
    }

    match task_kind {
        SemanticPassTaskKind::ManuscriptEntityConsolidation => {
            !matches!(surface, "Yeah" | "How" | "Three")
        }
        SemanticPassTaskKind::ReferenceKnowledgeConsolidation => {
            !matches!(surface, "Only" | "Why")
        }
        SemanticPassTaskKind::NotePlanningConsolidation => {
            !matches!(surface, "All" | "Every")
        }
    }
}

fn is_single_ascii_uppercase(surface: &str) -> bool {
    let mut chars = surface.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) => ch.is_ascii_uppercase(),
        _ => false,
    }
}

fn representative_snippets(cluster: &MentionCluster) -> Vec<String> {
    let mut snippets = Vec::new();

    for occurrence in &cluster.occurrences {
        if !snippets.contains(&occurrence.snippet) {
            snippets.push(occurrence.snippet.clone());
        }

        if snippets.len() == 2 {
            break;
        }
    }

    snippets
}

fn build_prompt(
    document_path: &str,
    document_type: DocumentType,
    archetype: DocumentArchetype,
    bundle_shape: SemanticBundleShape,
    task_kind: SemanticPassTaskKind,
    clusters: &[LabClusterRecord],
    structured_fields: &[LabFieldRecord],
    definitions: &[LabDefinitionRecord],
    section_summary_seeds: &[LabSummarySeedRecord],
    output_contract: &str,
) -> String {
    let mut prompt = String::new();
    let prompt_field_limit = prompt_field_limit(task_kind);
    let displayed_fields = structured_fields
        .iter()
        .take(prompt_field_limit)
        .collect::<Vec<_>>();
    let displayed_definitions = definitions
        .iter()
        .take(MAX_PROMPT_DEFINITIONS)
        .collect::<Vec<_>>();
    let displayed_summary_seeds = section_summary_seeds
        .iter()
        .take(MAX_PROMPT_SUMMARY_SEEDS)
        .collect::<Vec<_>>();

    prompt.push_str("You are running an experimental semantic consolidation pass.\n");
    prompt.push_str("Use only the provided evidence bundle. Do not invent canon.\n");
    prompt.push_str("If evidence is ambiguous or noisy, reject it or record an open question.\n\n");
    prompt.push_str(&format!("document_path: {document_path}\n"));
    prompt.push_str(&format!("document_type: {:?}\n", document_type));
    prompt.push_str(&format!("document_archetype: {:?}\n", archetype));
    prompt.push_str(&format!("bundle_shape: {:?}\n", bundle_shape));
    prompt.push_str(&format!("semantic_task_kind: {:?}\n\n", task_kind));

    prompt.push_str(task_instructions(task_kind));
    prompt.push_str("\n\n");

    prompt.push_str("Evidence bundle:\n");
    prompt.push_str(&format!("- mention_clusters: {}\n", clusters.len()));
    prompt.push_str(&format!(
        "- structured_fields: {} (showing {})\n",
        structured_fields.len(),
        displayed_fields.len()
    ));
    prompt.push_str(&format!(
        "- definitions: {} (showing {})\n",
        definitions.len(),
        displayed_definitions.len()
    ));
    prompt.push_str(&format!(
        "- section_summary_seeds: {} (showing {})\n\n",
        section_summary_seeds.len(),
        displayed_summary_seeds.len()
    ));

    prompt.push_str("Mention clusters:\n");
    for cluster in clusters {
        prompt.push_str(&format!(
            "- id={} surface={} occurrences={} features={:?}\n",
            cluster.id, cluster.display_surface, cluster.occurrence_count, cluster.aggregate_features
        ));
        if !cluster.member_surfaces.is_empty() {
            prompt.push_str(&format!("  members={:?}\n", cluster.member_surfaces));
        }
        if !cluster.linked_evidence_summaries.is_empty() {
            let displayed_links = cluster
                .linked_evidence_summaries
                .iter()
                .take(MAX_PROMPT_LINKED_EVIDENCE_PER_CLUSTER)
                .map(|summary| truncate_for_prompt(summary, MAX_PROMPT_TEXT_CHARS))
                .collect::<Vec<_>>();
            let omitted_links =
                cluster.linked_evidence_summaries.len().saturating_sub(displayed_links.len());
            prompt.push_str(&format!("  linked_evidence={displayed_links:?}\n"));
            if omitted_links > 0 {
                prompt.push_str(&format!("  linked_evidence_omitted={omitted_links}\n"));
            }
        }
        for snippet in cluster
            .representative_snippets
            .iter()
            .take(MAX_PROMPT_SNIPPETS_PER_CLUSTER)
        {
            prompt.push_str(&format!(
                "  snippet={}\n",
                truncate_for_prompt(snippet, MAX_PROMPT_TEXT_CHARS)
            ));
        }
    }

    prompt.push_str("\nStructured fields:\n");
    for field in &displayed_fields {
        prompt.push_str(&format!(
            "- id={} {}: {}\n",
            field.id,
            truncate_for_prompt(&field.label, 80),
            truncate_for_prompt(&field.value, MAX_PROMPT_TEXT_CHARS)
        ));
    }
    append_omitted_count(&mut prompt, structured_fields.len(), displayed_fields.len(), "structured_fields");

    prompt.push_str("\nDefinitions:\n");
    for definition in &displayed_definitions {
        prompt.push_str(&format!(
            "- id={} {} => {}\n",
            definition.id,
            truncate_for_prompt(&definition.term, 80),
            truncate_for_prompt(&definition.definition, MAX_PROMPT_TEXT_CHARS)
        ));
    }
    append_omitted_count(&mut prompt, definitions.len(), displayed_definitions.len(), "definitions");

    prompt.push_str("\nSection summary seeds:\n");
    for seed in &displayed_summary_seeds {
        prompt.push_str(&format!(
            "- id={} {} => {}\n",
            seed.id,
            truncate_for_prompt(&seed.scope, 80),
            truncate_for_prompt(&seed.text, MAX_PROMPT_TEXT_CHARS)
        ));
    }
    append_omitted_count(
        &mut prompt,
        section_summary_seeds.len(),
        displayed_summary_seeds.len(),
        "section_summary_seeds",
    );

    prompt.push_str("\nReturn only JSON matching this contract:\n");
    prompt.push_str(output_contract);
    prompt
}

fn prompt_field_limit(task_kind: SemanticPassTaskKind) -> usize {
    match task_kind {
        SemanticPassTaskKind::ManuscriptEntityConsolidation => MAX_PROMPT_MANUSCRIPT_FIELDS,
        SemanticPassTaskKind::ReferenceKnowledgeConsolidation => MAX_PROMPT_REFERENCE_FIELDS,
        SemanticPassTaskKind::NotePlanningConsolidation => MAX_PROMPT_NOTE_FIELDS,
    }
}

fn append_omitted_count(prompt: &mut String, total: usize, shown: usize, label: &str) {
    let omitted = total.saturating_sub(shown);
    if omitted > 0 {
        prompt.push_str(&format!("... omitted_{label}={omitted}\n"));
    }
}

fn truncate_for_prompt(text: &str, limit: usize) -> String {
    let truncated = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        format!("{truncated}...")
    } else {
        truncated
    }
}

pub fn semantic_output_schema(task_kind: SemanticPassTaskKind) -> Value {
    let mut properties = serde_json::Map::new();
    properties.insert("proposed_entities".to_string(), entity_array_schema());
    properties.insert(
        "candidate_entities_needing_review".to_string(),
        review_entity_array_schema(),
    );
    properties.insert("proposed_relationships".to_string(), relationship_array_schema());
    properties.insert("rejected_evidence".to_string(), rejected_array_schema());
    properties.insert("open_questions".to_string(), string_array_schema());

    let mut required = vec![
        "proposed_entities",
        "candidate_entities_needing_review",
        "proposed_relationships",
        "rejected_evidence",
        "open_questions",
    ];

    if !matches!(task_kind, SemanticPassTaskKind::ManuscriptEntityConsolidation) {
        properties.insert("proposed_terminology".to_string(), terminology_array_schema());
        required.push("proposed_terminology");
    }

    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": properties,
        "required": required,
    })
}

fn semantic_output_contract(task_kind: SemanticPassTaskKind) -> String {
    serde_json::to_string_pretty(&semantic_output_schema(task_kind))
        .expect("semantic output schema should serialize")
}

pub fn semantic_pass_task_kind(
    document_type: DocumentType,
    _archetype: DocumentArchetype,
) -> SemanticPassTaskKind {
    match document_type {
        DocumentType::Manuscript => SemanticPassTaskKind::ManuscriptEntityConsolidation,
        DocumentType::Reference => SemanticPassTaskKind::ReferenceKnowledgeConsolidation,
        DocumentType::Note => SemanticPassTaskKind::NotePlanningConsolidation,
    }
}

fn task_instructions(task_kind: SemanticPassTaskKind) -> &'static str {
    match task_kind {
        SemanticPassTaskKind::ManuscriptEntityConsolidation => {
            "Task rules:\n- Prefer untitled base names as canonical_name when both titled and untitled forms exist.\n- Do not promote title-only or role-only references such as generic rank words into proposed_entities.\n- Put ambiguous title references, one-off weak mentions, and thin list mentions into candidate_entities_needing_review or rejected_evidence.\n- Do not define terminology unless the provided evidence explicitly defines it.\n- Do not infer chronology, succession, or backstory unless directly stated in the evidence.\n- If a relationship is plausible but not explicit, put it in open_questions instead of proposed_relationships."
        }
        SemanticPassTaskKind::ReferenceKnowledgeConsolidation => {
            "Task rules:\n- Focus on explicit terminology, institutions, places, and system concepts.\n- Treat single-letter Latin symbols, section labels, and abbreviations as noise unless the evidence explicitly defines them.\n- Do not import outside genre knowledge to fill in missing definitions.\n- Put underspecified abbreviations and ambiguous entities into candidate_entities_needing_review or rejected_evidence.\n- Prefer proposed_terminology only when a definition is explicit or tightly paraphrasable from the supplied evidence."
        }
        SemanticPassTaskKind::NotePlanningConsolidation => {
            "Task rules:\n- Treat these notes as planning material, not settled canon.\n- Focus on participants, roles, goals, and planning relationships that are explicit in the note evidence.\n- Reject placeholders, template variables, and underspecified labels instead of promoting them.\n- Put plausible but underspecified participants into candidate_entities_needing_review.\n- Use open_questions to surface unresolved planning assumptions."
        }
    }
}

fn string_array_schema() -> Value {
    json!({
        "type": "array",
        "items": { "type": "string" }
    })
}

fn entity_array_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "canonical_name": { "type": "string" },
                "candidate_type": {
                    "type": "string",
                    "enum": ["person", "place", "ship", "institution", "concept", "unknown"]
                },
                "aliases": string_array_schema(),
                "supporting_evidence_ids": string_array_schema(),
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"]
                },
                "notes": { "type": "string" }
            },
            "required": [
                "canonical_name",
                "candidate_type",
                "aliases",
                "supporting_evidence_ids",
                "confidence",
                "notes"
            ]
        }
    })
}

fn review_entity_array_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "surface": { "type": "string" },
                "likely_type": {
                    "type": "string",
                    "enum": ["person", "place", "ship", "institution", "concept", "unknown"]
                },
                "supporting_evidence_ids": string_array_schema(),
                "review_reason": { "type": "string" },
                "notes": { "type": "string" }
            },
            "required": [
                "surface",
                "likely_type",
                "supporting_evidence_ids",
                "review_reason",
                "notes"
            ]
        }
    })
}

fn relationship_array_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "left_entity": { "type": "string" },
                "relationship": { "type": "string" },
                "right_entity": { "type": "string" },
                "supporting_evidence_ids": string_array_schema(),
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"]
                },
                "notes": { "type": "string" }
            },
            "required": [
                "left_entity",
                "relationship",
                "right_entity",
                "supporting_evidence_ids",
                "confidence",
                "notes"
            ]
        }
    })
}

fn terminology_array_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "term": { "type": "string" },
                "definition": { "type": "string" },
                "supporting_evidence_ids": string_array_schema(),
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"]
                },
                "notes": { "type": "string" }
            },
            "required": [
                "term",
                "definition",
                "supporting_evidence_ids",
                "confidence",
                "notes"
            ]
        }
    })
}

fn rejected_array_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "evidence_id": { "type": "string" },
                "reason": { "type": "string" }
            },
            "required": ["evidence_id", "reason"]
        }
    })
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;
    use writing_assist_core::{
        DocumentArchetype, DocumentType, MemorySourceReference, MentionCluster,
        MentionClusterLink, MentionClusterLinkKind, MentionFeature, MentionOccurrence,
        SentenceType, StructuredFieldCandidate, TargetAnchor,
    };

    use super::{
        build_semantic_pass_lab_packet, semantic_output_schema, SemanticBundleShape,
        SemanticPassTaskKind,
    };

    #[test]
    fn curated_bundle_filters_out_weak_singleton_clusters() {
        let strong = cluster(
            "Radiant Firth",
            vec![MentionFeature::MultiWord, MentionFeature::Repeated],
            2,
            vec![],
        );
        let weak = cluster("Childhood", vec![], 1, vec![]);

        let packet = build_semantic_pass_lab_packet(
            "semantic-pass-lab",
            "examples/reference.txt",
            DocumentType::Reference,
            DocumentArchetype::LooseNote,
            SemanticBundleShape::Curated,
            &[strong.clone(), weak.clone()],
            &[],
            &[],
            &[],
        );

        assert_eq!(packet.clusters.len(), 1);
        assert_eq!(packet.clusters[0].display_surface, strong.display_surface);
    }

    #[test]
    fn curated_bundle_filters_single_letter_latin_noise() {
        let meaningful = cluster(
            "Radiant Firth",
            vec![MentionFeature::MultiWord, MentionFeature::Repeated],
            2,
            vec![],
        );
        let noisy = cluster("C", vec![MentionFeature::Repeated], 6, vec![]);

        let packet = build_semantic_pass_lab_packet(
            "semantic-pass-lab",
            "examples/world.txt",
            DocumentType::Reference,
            DocumentArchetype::TaxonomyReference,
            SemanticBundleShape::Curated,
            &[meaningful.clone(), noisy],
            &[],
            &[],
            &[],
        );

        assert_eq!(packet.clusters.len(), 1);
        assert_eq!(packet.clusters[0].display_surface, meaningful.display_surface);
    }

    #[test]
    fn high_recall_bundle_keeps_weaker_clusters_for_comparison() {
        let strong = cluster(
            "Radiant Firth",
            vec![MentionFeature::MultiWord, MentionFeature::Repeated],
            2,
            vec![],
        );
        let weak = cluster("Childhood", vec![], 1, vec![]);

        let packet = build_semantic_pass_lab_packet(
            "semantic-pass-lab",
            "examples/reference.txt",
            DocumentType::Reference,
            DocumentArchetype::LooseNote,
            SemanticBundleShape::HighRecall,
            &[strong, weak.clone()],
            &[],
            &[],
            &[],
        );

        assert!(packet
            .clusters
            .iter()
            .any(|cluster| cluster.display_surface == weak.display_surface));
    }

    #[test]
    fn curated_bundle_keeps_linked_supporting_evidence_only() {
        let field = structured_field("role", "captain");
        let linked_cluster = cluster(
            "Yō",
            vec![],
            1,
            vec![MentionClusterLink {
                kind: MentionClusterLinkKind::StructuredField,
                evidence_id: field.id,
                summary: "role: captain".to_string(),
            }],
        );
        let unrelated_field = structured_field("location", "bridge");

        let packet = build_semantic_pass_lab_packet(
            "semantic-pass-lab",
            "examples/reference.txt",
            DocumentType::Reference,
            DocumentArchetype::DossierProfile,
            SemanticBundleShape::Curated,
            &[linked_cluster],
            &[field.clone(), unrelated_field],
            &[],
            &[],
        );

        assert_eq!(packet.structured_fields.len(), 1);
        assert_eq!(packet.structured_fields[0].id, field.id.to_string());
    }

    #[test]
    fn prompt_includes_output_contract_and_cluster_snippets() {
        let cluster = cluster(
            "Radiant Firth",
            vec![MentionFeature::MultiWord, MentionFeature::Repeated],
            2,
            vec![],
        );

        let packet = build_semantic_pass_lab_packet(
            "semantic-pass-lab",
            "examples/1. Radiant Firth.md",
            DocumentType::Manuscript,
            DocumentArchetype::Manuscript,
            SemanticBundleShape::Curated,
            &[cluster],
            &[],
            &[],
            &[],
        );

        assert!(packet.prompt.contains("Return only JSON matching this contract"));
        assert!(packet.prompt.contains("snippet=Snippet 1 for Radiant Firth"));
        assert!(packet.output_contract.contains("\"proposed_entities\""));
        assert!(packet
            .prompt
            .contains("Prefer untitled base names as canonical_name"));
        assert!(packet
            .output_contract
            .contains("\"candidate_entities_needing_review\""));
    }

    #[test]
    fn prompt_caps_linked_evidence_and_reports_omitted_count() {
        let cluster = cluster(
            "Scene",
            vec![MentionFeature::Repeated],
            2,
            (0..7)
                .map(|index| MentionClusterLink {
                    kind: MentionClusterLinkKind::StructuredField,
                    evidence_id: Uuid::new_v4(),
                    summary: format!("linked summary {}", index + 1),
                })
                .collect(),
        );

        let packet = build_semantic_pass_lab_packet(
            "semantic-pass-lab",
            "examples/briefing.txt",
            DocumentType::Note,
            DocumentArchetype::StoryPlanning,
            SemanticBundleShape::Curated,
            &[cluster],
            &[],
            &[],
            &[],
        );

        assert!(packet.prompt.contains("linked_evidence_omitted=2"));
        assert!(!packet.prompt.contains("linked summary 6"));
        assert!(!packet.prompt.contains("linked summary 7"));
    }

    #[test]
    fn note_prompt_caps_structured_fields_and_reports_omitted_count() {
        let fields = (0..35)
            .map(|index| structured_field(&format!("label {}", index + 1), "value"))
            .collect::<Vec<_>>();
        let links = fields
            .iter()
            .map(|field| MentionClusterLink {
                kind: MentionClusterLinkKind::StructuredField,
                evidence_id: field.id,
                summary: format!("{}: {}", field.label, field.value),
            })
            .collect::<Vec<_>>();
        let cluster = cluster("Scene", vec![MentionFeature::Repeated], 2, links);

        let packet = build_semantic_pass_lab_packet(
            "semantic-pass-lab",
            "examples/briefing.txt",
            DocumentType::Note,
            DocumentArchetype::StoryPlanning,
            SemanticBundleShape::Curated,
            &[cluster],
            &fields,
            &[],
            &[],
        );

        assert!(packet.prompt.contains("structured_fields: 35 (showing 30)"));
        assert!(packet.prompt.contains("... omitted_structured_fields=5"));
        assert!(!packet.prompt.contains("label 35"));
    }

    #[test]
    fn manuscript_schema_excludes_terminology_bucket() {
        let schema = semantic_output_schema(SemanticPassTaskKind::ManuscriptEntityConsolidation);

        assert!(schema
            .get("properties")
            .and_then(|value| value.get("proposed_terminology"))
            .is_none());
    }

    #[test]
    fn reference_schema_keeps_terminology_bucket() {
        let schema = semantic_output_schema(SemanticPassTaskKind::ReferenceKnowledgeConsolidation);

        assert!(schema
            .get("properties")
            .and_then(|value| value.get("proposed_terminology"))
            .is_some());
    }

    fn cluster(
        display_surface: &str,
        features: Vec<MentionFeature>,
        occurrence_count: usize,
        links: Vec<MentionClusterLink>,
    ) -> MentionCluster {
        let id = Uuid::new_v4();
        MentionCluster {
            id,
            display_surface: display_surface.to_string(),
            normalized_surface: display_surface.to_lowercase(),
            source: source_reference(),
            member_mention_ids: vec![Uuid::new_v4()],
            member_surfaces: vec![display_surface.to_string()],
            occurrences: (0..occurrence_count)
                .map(|index| MentionOccurrence {
                    span_anchor: TargetAnchor::span(index),
                    section_anchor: Some(TargetAnchor::section(0)),
                    heading: Some("Heading".to_string()),
                    snippet: format!("Snippet {} for {display_surface}", index + 1),
                    sentence_type: SentenceType::Narrative,
                    cooccurring_mentions: vec![],
                })
                .collect(),
            aggregate_features: features,
            linked_evidence: links,
            archetype: DocumentArchetype::Manuscript,
        }
    }

    fn structured_field(label: &str, value: &str) -> StructuredFieldCandidate {
        StructuredFieldCandidate {
            id: Uuid::new_v4(),
            label: label.to_string(),
            value: value.to_string(),
            source: source_reference(),
            contexts: vec![],
            archetype: DocumentArchetype::DossierProfile,
        }
    }

    fn source_reference() -> MemorySourceReference {
        MemorySourceReference::new(
            "examples/file.txt",
            vec![TargetAnchor::span(0)],
            0,
            10,
        )
    }
}
