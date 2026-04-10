use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::TargetAnchor;

/// Review state is the human approval gate for machine-derived memory.
///
/// Phase 3 deliberately keeps generated entities, facts, and summaries out of
/// task context until the user approves them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryReviewState {
    Pending,
    Approved,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStalenessState {
    Current,
    Stale,
}

/// Keeps every memory candidate tied back to the manuscript/reference text that
/// produced it, so later review and retrieval UI can show evidence rather than
/// orphaned facts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemorySourceReference {
    pub document_path: String,
    pub anchors: Vec<TargetAnchor>,
    pub start_char: usize,
    pub end_char: usize,
}

impl MemorySourceReference {
    pub fn new(
        document_path: impl Into<String>,
        anchors: Vec<TargetAnchor>,
        start_char: usize,
        end_char: usize,
    ) -> Self {
        Self {
            document_path: document_path.into(),
            anchors,
            start_char,
            end_char,
        }
    }
}

fn review_state_allows_reuse(review_state: &MemoryReviewState) -> bool {
    matches!(review_state, MemoryReviewState::Approved)
}

fn staleness_state_allows_reuse(staleness_state: &MemoryStalenessState) -> bool {
    matches!(staleness_state, MemoryStalenessState::Current)
}

fn reviewable_memory_is_reusable(
    review_state: &MemoryReviewState,
    staleness_state: &MemoryStalenessState,
) -> bool {
    // A record must pass both gates before retrieval/task context can reuse it:
    // the user approved it, and its source text has not changed underneath it.
    review_state_allows_reuse(review_state) && staleness_state_allows_reuse(staleness_state)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntityCandidate {
    pub id: Uuid,
    pub name: String,
    pub source: MemorySourceReference,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}

impl EntityCandidate {
    pub fn new(
        id: Uuid,
        name: impl Into<String>,
        source: MemorySourceReference,
        review_state: MemoryReviewState,
        staleness_state: MemoryStalenessState,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            source,
            review_state,
            staleness_state,
        }
    }

    pub fn is_reusable(&self) -> bool {
        reviewable_memory_is_reusable(&self.review_state, &self.staleness_state)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewableFact {
    pub id: Uuid,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub source: MemorySourceReference,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}

impl ReviewableFact {
    pub fn new(
        id: Uuid,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        source: MemorySourceReference,
        review_state: MemoryReviewState,
        staleness_state: MemoryStalenessState,
    ) -> Self {
        Self {
            id,
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            source,
            review_state,
            staleness_state,
        }
    }

    pub fn is_reusable(&self) -> bool {
        reviewable_memory_is_reusable(&self.review_state, &self.staleness_state)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewableSummary {
    pub id: Uuid,
    pub scope: String,
    pub text: String,
    pub source: MemorySourceReference,
    pub review_state: MemoryReviewState,
    pub staleness_state: MemoryStalenessState,
}

impl ReviewableSummary {
    pub fn new(
        id: Uuid,
        scope: impl Into<String>,
        text: impl Into<String>,
        source: MemorySourceReference,
        review_state: MemoryReviewState,
        staleness_state: MemoryStalenessState,
    ) -> Self {
        Self {
            id,
            scope: scope.into(),
            text: text.into(),
            source,
            review_state,
            staleness_state,
        }
    }

    pub fn is_reusable(&self) -> bool {
        reviewable_memory_is_reusable(&self.review_state, &self.staleness_state)
    }
}
