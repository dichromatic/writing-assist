use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{DocumentArchetype, MemorySourceReference};

/// Compact deterministic evidence selected for retrieval and later semantic
/// consolidation.
///
/// These records are not approved memory and are not final semantic claims.
/// They are a smaller, source-linked view over raw harvested evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromotedEvidenceBundle {
    pub promoted: Vec<PromotedEvidenceCandidate>,
    pub review_only: Vec<PromotedEvidenceCandidate>,
    pub suppressed: Vec<SuppressedEvidenceCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromotedEvidenceCandidate {
    pub id: Uuid,
    pub display_surface: String,
    pub normalized_surface: String,
    pub kind: PromotedEvidenceKind,
    pub source: MemorySourceReference,
    pub archetype: DocumentArchetype,
    pub evidence_ids: Vec<Uuid>,
    pub reasons: Vec<PromotionReason>,
    pub snippets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromotedEvidenceKind {
    EntityLike,
    Terminology,
    FieldBackedContext,
    ReviewOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromotionReason {
    RepeatedMention,
    MultiWordMention,
    TitledMention,
    DefinitionBacked,
    FieldBacked,
    LexiconBacked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SuppressedEvidenceCandidate {
    pub evidence_id: Uuid,
    pub display_surface: String,
    pub normalized_surface: String,
    pub source: MemorySourceReference,
    pub archetype: DocumentArchetype,
    pub reason: EvidenceSuppressionReason,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSuppressionReason {
    WeakSingleton,
    UnresolvedAbbreviation,
    RelationshipLikeFieldNotPromoted,
}
