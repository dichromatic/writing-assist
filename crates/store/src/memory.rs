use std::path::Path;

use sqlx::sqlite::SqliteRow;
use sqlx::Row;
use uuid::Uuid;
use writing_assist_core::{
    EntityCandidate, MemoryReviewState, MemorySourceReference, MemoryStalenessState,
    ReviewableFact, ReviewableSummary, TargetAnchor,
};

use crate::{open_project_database, StoreError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredMemoryKind {
    Entity,
    Fact,
    Summary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRecordFilter {
    All,
    Pending,
    Approved,
    Rejected,
    Stale,
    Reusable,
}

impl MemoryRecordFilter {
    fn where_clause(self) -> &'static str {
        match self {
            Self::All => "",
            Self::Pending => "WHERE review_state = 'pending'",
            Self::Approved => "WHERE review_state = 'approved'",
            Self::Rejected => "WHERE review_state = 'rejected'",
            Self::Stale => "WHERE staleness_state = 'stale'",
            Self::Reusable => "WHERE review_state = 'approved' AND staleness_state = 'current'",
        }
    }
}

fn table_for_kind(kind: StoredMemoryKind) -> &'static str {
    match kind {
        StoredMemoryKind::Entity => "entity_candidates",
        StoredMemoryKind::Fact => "reviewable_facts",
        StoredMemoryKind::Summary => "reviewable_summaries",
    }
}

fn review_state_to_storage_value(review_state: &MemoryReviewState) -> &'static str {
    match review_state {
        MemoryReviewState::Pending => "pending",
        MemoryReviewState::Approved => "approved",
        MemoryReviewState::Rejected => "rejected",
    }
}

fn review_state_from_storage_value(value: &str) -> Result<MemoryReviewState, StoreError> {
    match value {
        "pending" => Ok(MemoryReviewState::Pending),
        "approved" => Ok(MemoryReviewState::Approved),
        "rejected" => Ok(MemoryReviewState::Rejected),
        _ => Err(StoreError::InvalidStoredMemoryReviewState(value.to_string())),
    }
}

fn staleness_state_to_storage_value(staleness_state: &MemoryStalenessState) -> &'static str {
    match staleness_state {
        MemoryStalenessState::Current => "current",
        MemoryStalenessState::Stale => "stale",
    }
}

fn staleness_state_from_storage_value(value: &str) -> Result<MemoryStalenessState, StoreError> {
    match value {
        "current" => Ok(MemoryStalenessState::Current),
        "stale" => Ok(MemoryStalenessState::Stale),
        _ => Err(StoreError::InvalidStoredMemoryStalenessState(value.to_string())),
    }
}

fn source_reference_from_row(row: &SqliteRow) -> Result<MemorySourceReference, StoreError> {
    let anchors_json: String = row.get("source_anchors_json");
    let anchors: Vec<TargetAnchor> = serde_json::from_str(&anchors_json)?;
    let start_char: i64 = row.get("source_start_char");
    let end_char: i64 = row.get("source_end_char");

    Ok(MemorySourceReference::new(
        row.get::<String, _>("source_document_path"),
        anchors,
        start_char as usize,
        end_char as usize,
    ))
}

pub async fn save_entity_candidates(
    project_root: &Path,
    candidates: &[EntityCandidate],
) -> Result<(), StoreError> {
    let pool = open_project_database(project_root, true)
        .await?
        .expect("create_if_missing=true should return a pool");
    let mut transaction = pool.begin().await?;

    for candidate in candidates {
        // Candidate IDs are stable fields from core/index.
        // Upserting by ID gives deterministic duplicate handling.
        sqlx::query(
            r#"
            INSERT INTO entity_candidates (
                id, name, source_document_path, source_anchors_json,
                source_start_char, source_end_char, review_state, staleness_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                source_document_path = excluded.source_document_path,
                source_anchors_json = excluded.source_anchors_json,
                source_start_char = excluded.source_start_char,
                source_end_char = excluded.source_end_char,
                review_state = excluded.review_state,
                staleness_state = excluded.staleness_state
            "#,
        )
        .bind(candidate.id.to_string())
        .bind(&candidate.name)
        .bind(&candidate.source.document_path)
        .bind(serde_json::to_string(&candidate.source.anchors)?)
        .bind(candidate.source.start_char as i64)
        .bind(candidate.source.end_char as i64)
        .bind(review_state_to_storage_value(&candidate.review_state))
        .bind(staleness_state_to_storage_value(&candidate.staleness_state))
        .execute(&mut *transaction)
        .await?;
    }

    transaction.commit().await?;

    Ok(())
}

pub async fn save_reviewable_facts(
    project_root: &Path,
    facts: &[ReviewableFact],
) -> Result<(), StoreError> {
    let pool = open_project_database(project_root, true)
        .await?
        .expect("create_if_missing=true should return a pool");
    let mut transaction = pool.begin().await?;

    for fact in facts {
        sqlx::query(
            r#"
            INSERT INTO reviewable_facts (
                id, subject, predicate, object, source_document_path, source_anchors_json,
                source_start_char, source_end_char, review_state, staleness_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                subject = excluded.subject,
                predicate = excluded.predicate,
                object = excluded.object,
                source_document_path = excluded.source_document_path,
                source_anchors_json = excluded.source_anchors_json,
                source_start_char = excluded.source_start_char,
                source_end_char = excluded.source_end_char,
                review_state = excluded.review_state,
                staleness_state = excluded.staleness_state
            "#,
        )
        .bind(fact.id.to_string())
        .bind(&fact.subject)
        .bind(&fact.predicate)
        .bind(&fact.object)
        .bind(&fact.source.document_path)
        .bind(serde_json::to_string(&fact.source.anchors)?)
        .bind(fact.source.start_char as i64)
        .bind(fact.source.end_char as i64)
        .bind(review_state_to_storage_value(&fact.review_state))
        .bind(staleness_state_to_storage_value(&fact.staleness_state))
        .execute(&mut *transaction)
        .await?;
    }

    transaction.commit().await?;

    Ok(())
}

pub async fn save_reviewable_summaries(
    project_root: &Path,
    summaries: &[ReviewableSummary],
) -> Result<(), StoreError> {
    let pool = open_project_database(project_root, true)
        .await?
        .expect("create_if_missing=true should return a pool");
    let mut transaction = pool.begin().await?;

    for summary in summaries {
        sqlx::query(
            r#"
            INSERT INTO reviewable_summaries (
                id, scope, text, source_document_path, source_anchors_json,
                source_start_char, source_end_char, review_state, staleness_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                scope = excluded.scope,
                text = excluded.text,
                source_document_path = excluded.source_document_path,
                source_anchors_json = excluded.source_anchors_json,
                source_start_char = excluded.source_start_char,
                source_end_char = excluded.source_end_char,
                review_state = excluded.review_state,
                staleness_state = excluded.staleness_state
            "#,
        )
        .bind(summary.id.to_string())
        .bind(&summary.scope)
        .bind(&summary.text)
        .bind(&summary.source.document_path)
        .bind(serde_json::to_string(&summary.source.anchors)?)
        .bind(summary.source.start_char as i64)
        .bind(summary.source.end_char as i64)
        .bind(review_state_to_storage_value(&summary.review_state))
        .bind(staleness_state_to_storage_value(&summary.staleness_state))
        .execute(&mut *transaction)
        .await?;
    }

    transaction.commit().await?;

    Ok(())
}

pub async fn list_entity_candidates(
    project_root: &Path,
    filter: MemoryRecordFilter,
) -> Result<Vec<EntityCandidate>, StoreError> {
    let Some(pool) = open_project_database(project_root, false).await? else {
        return Ok(Vec::new());
    };
    let sql = format!(
        "SELECT * FROM entity_candidates {} ORDER BY rowid",
        filter.where_clause()
    );
    let rows = sqlx::query(&sql).fetch_all(&pool).await?;

    rows.into_iter()
        .map(|row| {
            Ok(EntityCandidate::new(
                Uuid::parse_str(&row.get::<String, _>("id"))?,
                row.get::<String, _>("name"),
                source_reference_from_row(&row)?,
                review_state_from_storage_value(&row.get::<String, _>("review_state"))?,
                staleness_state_from_storage_value(&row.get::<String, _>("staleness_state"))?,
            ))
        })
        .collect()
}

pub async fn list_reviewable_facts(
    project_root: &Path,
    filter: MemoryRecordFilter,
) -> Result<Vec<ReviewableFact>, StoreError> {
    let Some(pool) = open_project_database(project_root, false).await? else {
        return Ok(Vec::new());
    };
    let sql = format!(
        "SELECT * FROM reviewable_facts {} ORDER BY rowid",
        filter.where_clause()
    );
    let rows = sqlx::query(&sql).fetch_all(&pool).await?;

    rows.into_iter()
        .map(|row| {
            Ok(ReviewableFact::new(
                Uuid::parse_str(&row.get::<String, _>("id"))?,
                row.get::<String, _>("subject"),
                row.get::<String, _>("predicate"),
                row.get::<String, _>("object"),
                source_reference_from_row(&row)?,
                review_state_from_storage_value(&row.get::<String, _>("review_state"))?,
                staleness_state_from_storage_value(&row.get::<String, _>("staleness_state"))?,
            ))
        })
        .collect()
}

pub async fn list_reviewable_summaries(
    project_root: &Path,
    filter: MemoryRecordFilter,
) -> Result<Vec<ReviewableSummary>, StoreError> {
    let Some(pool) = open_project_database(project_root, false).await? else {
        return Ok(Vec::new());
    };
    let sql = format!(
        "SELECT * FROM reviewable_summaries {} ORDER BY rowid",
        filter.where_clause()
    );
    let rows = sqlx::query(&sql).fetch_all(&pool).await?;

    rows.into_iter()
        .map(|row| {
            Ok(ReviewableSummary::new(
                Uuid::parse_str(&row.get::<String, _>("id"))?,
                row.get::<String, _>("scope"),
                row.get::<String, _>("text"),
                source_reference_from_row(&row)?,
                review_state_from_storage_value(&row.get::<String, _>("review_state"))?,
                staleness_state_from_storage_value(&row.get::<String, _>("staleness_state"))?,
            ))
        })
        .collect()
}

pub async fn update_memory_review_state(
    project_root: &Path,
    kind: StoredMemoryKind,
    id: Uuid,
    review_state: MemoryReviewState,
) -> Result<(), StoreError> {
    let pool = open_project_database(project_root, true)
        .await?
        .expect("create_if_missing=true should return a pool");
    let sql = format!(
        "UPDATE {} SET review_state = ? WHERE id = ?",
        table_for_kind(kind)
    );
    let result = sqlx::query(&sql)
        .bind(review_state_to_storage_value(&review_state))
        .bind(id.to_string())
        .execute(&pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(StoreError::MemoryRecordNotFound { kind, id });
    }

    Ok(())
}

pub async fn mark_memory_stale_for_document(
    project_root: &Path,
    document_path: &str,
) -> Result<u64, StoreError> {
    let Some(pool) = open_project_database(project_root, false).await? else {
        return Ok(0);
    };
    let mut changed = 0;

    for table in [
        table_for_kind(StoredMemoryKind::Entity),
        table_for_kind(StoredMemoryKind::Fact),
        table_for_kind(StoredMemoryKind::Summary),
    ] {
        let sql = format!(
            "UPDATE {} SET staleness_state = 'stale' WHERE source_document_path = ?",
            table
        );
        changed += sqlx::query(&sql)
            .bind(document_path)
            .execute(&pool)
            .await?
            .rows_affected();
    }

    Ok(changed)
}
