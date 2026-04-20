use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{DocumentArchetype, MemorySourceReference};

/// Bootstrapped lexicon entries are deterministic extraction infrastructure.
///
/// They are intentionally seedless and review-agnostic: the goal is to retain
/// project-local lexical support for later harvesting and semantic
/// consolidation, not to create approved canon memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BootstrappedLexiconEntryKind {
    Character,
    Place,
    Faction,
    Artifact,
    Terminology,
    Unresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LexiconSupportRecordKind {
    MentionCluster,
    StructuredField,
    Definition,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LexiconBootstrapRule {
    RepeatedMention,
    TitledMention,
    AliasField,
    ParticipantField,
    RoleField,
    DefinitionTerm,
    LinkedStructuredField,
    LinkedDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LexiconSupportRecord {
    pub evidence_id: Uuid,
    pub kind: LexiconSupportRecordKind,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BootstrappedLexiconEntry {
    pub id: Uuid,
    pub canonical_surface: String,
    pub normalized_surface: String,
    pub kind: BootstrappedLexiconEntryKind,
    pub source: MemorySourceReference,
    pub occurrence_count: usize,
    pub archetypes_seen: Vec<DocumentArchetype>,
    pub rule_sources: Vec<LexiconBootstrapRule>,
    pub evidence: Vec<LexiconSupportRecord>,
}
