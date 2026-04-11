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
pub struct MentionCandidate {
    pub id: Uuid,
    pub surface: String,
    pub normalized_surface: String,
    pub occurrence_count: usize,
    pub source: MemorySourceReference,
    pub contexts: Vec<EvidenceContext>,
    pub features: Vec<MentionFeature>,
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
