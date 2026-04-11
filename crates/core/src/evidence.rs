use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{DocumentArchetype, MemorySourceReference, TargetAnchor};

/// Evidence records are deterministic, source-linked harvests that sit before
/// provider-backed semantic consolidation. They are intentionally descriptive:
/// they preserve what the parser/indexer saw without claiming final truth.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceContext {
    pub span_anchor: TargetAnchor,
    pub section_anchor: Option<TargetAnchor>,
    pub heading: Option<String>,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MentionFeature {
    Repeated,
    MultiWord,
    Titled,
    HeadingMentioned,
    PossessiveObserved,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SentenceType {
    Dialogue,
    Narrative,
    Heading,
    ListItem,
    BlockQuote,
    Other,
}

/// Per-occurrence local evidence for a harvested mention.
///
/// This lets later semantic consolidation judge a mention without reloading the
/// full span store for every candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MentionOccurrence {
    pub span_anchor: TargetAnchor,
    pub section_anchor: Option<TargetAnchor>,
    pub heading: Option<String>,
    pub snippet: String,
    pub sentence_type: SentenceType,
    pub cooccurring_mentions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MentionClusterLinkKind {
    StructuredField,
    Definition,
    SectionSummarySeed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MentionClusterLink {
    pub kind: MentionClusterLinkKind,
    pub evidence_id: Uuid,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MentionCandidate {
    pub id: Uuid,
    pub surface: String,
    pub normalized_surface: String,
    pub source: MemorySourceReference,
    pub occurrences: Vec<MentionOccurrence>,
    pub aggregate_features: Vec<MentionFeature>,
    pub archetype: DocumentArchetype,
}

/// Deterministic per-document grouping over harvested mentions.
///
/// Clusters stay local to one document and deliberately stop short of semantic
/// identity across files. Their job is to make the later semantic layer consume
/// grouped evidence instead of isolated surface forms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MentionCluster {
    pub id: Uuid,
    pub display_surface: String,
    pub normalized_surface: String,
    pub source: MemorySourceReference,
    pub member_mention_ids: Vec<Uuid>,
    pub member_surfaces: Vec<String>,
    pub occurrences: Vec<MentionOccurrence>,
    pub aggregate_features: Vec<MentionFeature>,
    pub linked_evidence: Vec<MentionClusterLink>,
    pub archetype: DocumentArchetype,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructuredFieldCandidate {
    pub id: Uuid,
    pub label: String,
    pub value: String,
    pub source: MemorySourceReference,
    pub contexts: Vec<EvidenceContext>,
    pub archetype: DocumentArchetype,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DefinitionCandidate {
    pub id: Uuid,
    pub term: String,
    pub definition: String,
    pub source: MemorySourceReference,
    pub contexts: Vec<EvidenceContext>,
    pub archetype: DocumentArchetype,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SectionSummarySeed {
    pub id: Uuid,
    pub scope: String,
    pub text: String,
    pub source: MemorySourceReference,
    pub contexts: Vec<EvidenceContext>,
    pub archetype: DocumentArchetype,
}
