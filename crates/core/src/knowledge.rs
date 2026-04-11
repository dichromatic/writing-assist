use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{MemoryReviewState, MemorySourceReference, MemoryStalenessState};

/// High-level structural category for non-manuscript project documents.
///
/// This lets deterministic extraction and later LLM tasks choose schemas that
/// match the actual shape of a document instead of flattening everything into
/// generic facts or summaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentArchetype {
    Manuscript,
    DossierProfile,
    StoryPlanning,
    TaxonomyReference,
    ExpositoryWorldArticle,
    LooseNote,
}

/// Describes how a structured candidate is expected to be used in the product.
///
/// The key distinction is whether a record looks like canon/reference memory,
/// planning-only material, or temporary working context that should stay opt-in.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StructuredKnowledgeIntendedUse {
    CanonReference,
    PlanningOnly,
    WorkingContext,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StructuredKnowledgeCandidateKind {
    EntityProfile,
    Relationship,
    TimelineEvent,
    StoryArc,
    WorldRule,
    Terminology,
    ExtractiveSummary,
}

pub fn structured_knowledge_intended_use(
    archetype: DocumentArchetype,
    candidate_kind: StructuredKnowledgeCandidateKind,
) -> StructuredKnowledgeIntendedUse {
    match (archetype, candidate_kind) {
        (DocumentArchetype::StoryPlanning, _) => StructuredKnowledgeIntendedUse::PlanningOnly,
        (DocumentArchetype::LooseNote, _) => StructuredKnowledgeIntendedUse::WorkingContext,
        (DocumentArchetype::Manuscript, StructuredKnowledgeCandidateKind::ExtractiveSummary) => {
            StructuredKnowledgeIntendedUse::WorkingContext
        }
        (DocumentArchetype::Manuscript, _) => StructuredKnowledgeIntendedUse::CanonReference,
        (DocumentArchetype::DossierProfile, StructuredKnowledgeCandidateKind::StoryArc) => {
            StructuredKnowledgeIntendedUse::WorkingContext
        }
        (DocumentArchetype::DossierProfile, _) => StructuredKnowledgeIntendedUse::CanonReference,
        (DocumentArchetype::TaxonomyReference, _) => StructuredKnowledgeIntendedUse::CanonReference,
        (DocumentArchetype::ExpositoryWorldArticle, _) => {
            StructuredKnowledgeIntendedUse::CanonReference
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntityProfileCandidate {
    pub id: Uuid,
    pub name: String,
    pub role: Option<String>,
    pub traits: Vec<String>,
    pub details: Vec<String>,
    pub source: MemorySourceReference,
    pub intended_use: StructuredKnowledgeIntendedUse,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelationshipCandidate {
    pub id: Uuid,
    pub subject: String,
    pub relationship: String,
    pub object: String,
    pub details: Vec<String>,
    pub source: MemorySourceReference,
    pub intended_use: StructuredKnowledgeIntendedUse,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TimelineEventCandidate {
    pub id: Uuid,
    pub label: String,
    pub timeframe: Option<String>,
    pub outcome: Option<String>,
    pub details: Vec<String>,
    pub source: MemorySourceReference,
    pub intended_use: StructuredKnowledgeIntendedUse,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoryArcCandidate {
    pub id: Uuid,
    pub title: String,
    pub scope: Option<String>,
    pub tone: Vec<String>,
    pub purpose: Vec<String>,
    pub beats: Vec<String>,
    pub outcome: Option<String>,
    pub source: MemorySourceReference,
    pub intended_use: StructuredKnowledgeIntendedUse,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldRuleCandidate {
    pub id: Uuid,
    pub subject: String,
    pub category: Option<String>,
    pub statement: String,
    pub details: Vec<String>,
    pub source: MemorySourceReference,
    pub intended_use: StructuredKnowledgeIntendedUse,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TerminologyCandidate {
    pub id: Uuid,
    pub term: String,
    pub category: Option<String>,
    pub definition: String,
    pub related_terms: Vec<String>,
    pub source: MemorySourceReference,
    pub intended_use: StructuredKnowledgeIntendedUse,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}
