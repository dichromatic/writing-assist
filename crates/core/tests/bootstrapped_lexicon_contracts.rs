use uuid::Uuid;
use writing_assist_core::{
    BootstrappedLexiconEntry, BootstrappedLexiconEntryKind, DocumentArchetype,
    LexiconBootstrapRule, LexiconSupportRecord, LexiconSupportRecordKind, MemorySourceReference,
    TargetAnchor,
};

#[test]
fn bootstrapped_lexicon_entries_serialize_with_kind_and_rule_provenance() {
    let entry = BootstrappedLexiconEntry {
        id: Uuid::nil(),
        canonical_surface: "Captain Mara".to_string(),
        normalized_surface: "captain mara".to_string(),
        kind: BootstrappedLexiconEntryKind::Character,
        source: MemorySourceReference::new(
            "story planning/harbor-profile.md",
            vec![TargetAnchor::span(0), TargetAnchor::span(2)],
            0,
            120,
        ),
        occurrence_count: 3,
        archetypes_seen: vec![DocumentArchetype::DossierProfile],
        rule_sources: vec![
            LexiconBootstrapRule::TitledMention,
            LexiconBootstrapRule::AliasField,
        ],
        evidence: vec![
            LexiconSupportRecord {
                evidence_id: Uuid::nil(),
                kind: LexiconSupportRecordKind::MentionCluster,
                summary: "Captain Mara / Mara".to_string(),
            },
            LexiconSupportRecord {
                evidence_id: Uuid::nil(),
                kind: LexiconSupportRecordKind::StructuredField,
                summary: "Alias: Mara".to_string(),
            },
        ],
    };

    let serialized = serde_json::to_value(&entry).expect("serialize bootstrapped lexicon entry");

    assert_eq!(serialized["kind"], "character");
    assert_eq!(serialized["occurrence_count"], 3);
    assert_eq!(serialized["rule_sources"][0], "titled_mention");
    assert_eq!(serialized["evidence"][1]["kind"], "structured_field");
}
