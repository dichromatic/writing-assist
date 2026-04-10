use uuid::Uuid;
use writing_assist_core::{
    EntityCandidate, MemoryReviewState, MemorySourceReference, MemoryStalenessState,
    ReviewableFact, ReviewableSummary, TargetAnchor,
};
use writing_assist_store::{
    list_entity_candidates, list_reviewable_facts, list_reviewable_summaries,
    mark_memory_stale_for_document, save_entity_candidates, save_reviewable_facts,
    save_reviewable_summaries, update_memory_review_state, MemoryRecordFilter,
    StoredMemoryKind,
};

fn source(document_path: &str) -> MemorySourceReference {
    MemorySourceReference::new(
        document_path,
        vec![TargetAnchor::span(2), TargetAnchor::section(1)],
        12,
        48,
    )
}

fn entity(id: u128, name: &str, document_path: &str) -> EntityCandidate {
    EntityCandidate::new(
        Uuid::from_u128(id),
        name,
        source(document_path),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    )
}

#[tokio::test]
async fn pending_memory_records_persist_and_reload_with_source_references() {
    let project_root = tempfile::tempdir().expect("project root");
    let entity = entity(1, "Radiant Firth", "chapters/chapter-1.md");
    let fact = ReviewableFact::new(
        Uuid::from_u128(2),
        "Radiant Firth",
        "located_in",
        "Outer shipping lane",
        source("world/ships.md"),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    );
    let summary = ReviewableSummary::new(
        Uuid::from_u128(3),
        "chapter",
        "The crew sights the Radiant Firth.",
        source("chapters/chapter-1.md"),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    );

    save_entity_candidates(project_root.path(), &[entity.clone()])
        .await
        .expect("save entity candidates");
    save_reviewable_facts(project_root.path(), &[fact.clone()])
        .await
        .expect("save facts");
    save_reviewable_summaries(project_root.path(), &[summary.clone()])
        .await
        .expect("save summaries");

    assert_eq!(
        list_entity_candidates(project_root.path(), MemoryRecordFilter::Pending)
            .await
            .expect("list entity candidates"),
        vec![entity]
    );
    assert_eq!(
        list_reviewable_facts(project_root.path(), MemoryRecordFilter::Pending)
            .await
            .expect("list facts"),
        vec![fact]
    );
    assert_eq!(
        list_reviewable_summaries(project_root.path(), MemoryRecordFilter::Pending)
            .await
            .expect("list summaries"),
        vec![summary]
    );
}

#[tokio::test]
async fn review_and_staleness_transitions_gate_reusable_memory() {
    let project_root = tempfile::tempdir().expect("project root");
    let approved = entity(10, "Ysolde", "chapters/chapter-1.md");
    let rejected = entity(11, "False Lead", "chapters/chapter-1.md");
    let stale = entity(12, "Old Port", "chapters/chapter-2.md");

    save_entity_candidates(
        project_root.path(),
        &[approved.clone(), rejected.clone(), stale.clone()],
    )
    .await
    .expect("save entity candidates");

    update_memory_review_state(
        project_root.path(),
        StoredMemoryKind::Entity,
        approved.id,
        MemoryReviewState::Approved,
    )
    .await
    .expect("approve entity");
    update_memory_review_state(
        project_root.path(),
        StoredMemoryKind::Entity,
        rejected.id,
        MemoryReviewState::Rejected,
    )
    .await
    .expect("reject entity");
    update_memory_review_state(
        project_root.path(),
        StoredMemoryKind::Entity,
        stale.id,
        MemoryReviewState::Approved,
    )
    .await
    .expect("approve stale entity before stale marker");
    mark_memory_stale_for_document(project_root.path(), "chapters/chapter-2.md")
        .await
        .expect("mark memory stale");

    let reusable = list_entity_candidates(project_root.path(), MemoryRecordFilter::Reusable)
        .await
        .expect("list reusable entities");
    let rejected = list_entity_candidates(project_root.path(), MemoryRecordFilter::Rejected)
        .await
        .expect("list rejected entities");
    let stale = list_entity_candidates(project_root.path(), MemoryRecordFilter::Stale)
        .await
        .expect("list stale entities");

    assert_eq!(
        reusable
            .iter()
            .map(|candidate| candidate.name.as_str())
            .collect::<Vec<_>>(),
        vec!["Ysolde"]
    );
    assert_eq!(
        rejected
            .iter()
            .map(|candidate| candidate.name.as_str())
            .collect::<Vec<_>>(),
        vec!["False Lead"]
    );
    assert_eq!(
        stale
            .iter()
            .map(|candidate| candidate.name.as_str())
            .collect::<Vec<_>>(),
        vec!["Old Port"]
    );
}

#[tokio::test]
async fn fact_and_summary_review_states_persist_through_reusable_queries() {
    let project_root = tempfile::tempdir().expect("project root");
    let fact = ReviewableFact::new(
        Uuid::from_u128(30),
        "Radiant Firth",
        "captain",
        "Ysolde",
        source("world/ships.md"),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    );
    let summary = ReviewableSummary::new(
        Uuid::from_u128(31),
        "chapter",
        "The old port record is superseded.",
        source("chapters/chapter-2.md"),
        MemoryReviewState::Pending,
        MemoryStalenessState::Current,
    );

    save_reviewable_facts(project_root.path(), &[fact.clone()])
        .await
        .expect("save fact");
    save_reviewable_summaries(project_root.path(), &[summary.clone()])
        .await
        .expect("save summary");
    update_memory_review_state(
        project_root.path(),
        StoredMemoryKind::Fact,
        fact.id,
        MemoryReviewState::Approved,
    )
    .await
    .expect("approve fact");
    update_memory_review_state(
        project_root.path(),
        StoredMemoryKind::Summary,
        summary.id,
        MemoryReviewState::Approved,
    )
    .await
    .expect("approve summary");
    mark_memory_stale_for_document(project_root.path(), "chapters/chapter-2.md")
        .await
        .expect("mark stale summary");

    let reusable_facts = list_reviewable_facts(project_root.path(), MemoryRecordFilter::Reusable)
        .await
        .expect("list reusable facts");
    let reusable_summaries =
        list_reviewable_summaries(project_root.path(), MemoryRecordFilter::Reusable)
            .await
            .expect("list reusable summaries");
    let stale_summaries =
        list_reviewable_summaries(project_root.path(), MemoryRecordFilter::Stale)
            .await
            .expect("list stale summaries");

    assert_eq!(
        reusable_facts
            .iter()
            .map(|record| record.subject.as_str())
            .collect::<Vec<_>>(),
        vec!["Radiant Firth"]
    );
    assert!(reusable_summaries.is_empty());
    assert_eq!(stale_summaries, vec![{
        let mut stale_summary = summary;
        stale_summary.review_state = MemoryReviewState::Approved;
        stale_summary.staleness_state = MemoryStalenessState::Stale;
        stale_summary
    }]);
}

#[tokio::test]
async fn duplicate_entity_candidates_upsert_deterministically_by_id() {
    let project_root = tempfile::tempdir().expect("project root");
    let first = entity(20, "Old Name", "chapters/chapter-1.md");
    let replacement = entity(20, "New Name", "chapters/chapter-1.md");

    save_entity_candidates(project_root.path(), &[first])
        .await
        .expect("initial save");
    save_entity_candidates(project_root.path(), &[replacement.clone()])
        .await
        .expect("replacement save");

    assert_eq!(
        list_entity_candidates(project_root.path(), MemoryRecordFilter::All)
            .await
            .expect("list entity candidates"),
        vec![replacement]
    );
}
