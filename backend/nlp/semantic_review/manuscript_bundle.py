"""
Manuscript review bundle helpers - package and render manuscript handoff data.

.. code-block:: mermaid

    flowchart TD
        A[DocumentEntityRecord list] --> B[Build ManuscriptReviewBundle]
        C[CorpusEntity list] --> B
        D[ReferenceCandidate list] --> B
        E[ReferenceCluster list] --> B
        F[ConflictRecord list] --> B
        G[CharacterSemanticSummary list] --> B
        H[ReviewTask list] --> B
        B --> I[JSON-safe manuscript artifact]
        B --> J[Human-readable manuscript report]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.nlp.text_filtering import strip_emoji, to_llm_safe_jsonable
from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictRecord,
    CorpusEntity,
    DocumentEntityBucket,
    LexiconCategory,
    ManuscriptReviewBundle,
    ReferenceCandidate,
    ReferenceCandidateType,
    ReferenceCluster,
)


def build_manuscript_review_bundle(
    paths: list[Path],
    entity_records,
    corpus_entities: list[CorpusEntity],
    reference_candidates: list[ReferenceCandidate],
    reference_clusters: list[ReferenceCluster],
    conflict_records: list[ConflictRecord],
    character_summaries: list[CharacterSemanticSummary],
    review_tasks,
) -> ManuscriptReviewBundle:
    """Build the persisted manuscript handoff artifact.

    Args:
        paths: Source documents included in the corpus run.
        entity_records: Document-local entity summaries.
        corpus_entities: Corpus-level canonical entities.
        reference_candidates: Raw deferred reference mentions.
        reference_clusters: Grouped reference review objects.
        conflict_records: Typed conflict records.
        character_summaries: Character-centric semantic summaries.
        review_tasks: Question-first semantic review tasks.

    Returns:
        Persisted manuscript handoff object that can drive both JSON and text
        outputs.
    """
    return ManuscriptReviewBundle(
        document_paths=[str(path) for path in paths],
        entity_records=list(entity_records),
        canonical_entities=list(corpus_entities),
        reference_candidates=list(reference_candidates),
        reference_clusters=list(reference_clusters),
        conflict_records=list(conflict_records),
        character_summaries=list(character_summaries),
        review_tasks=list(review_tasks),
    )


def manuscript_bundle_to_jsonable(bundle: ManuscriptReviewBundle) -> dict[str, Any]:
    """Convert a manuscript review bundle into JSON-safe nested data.

    Args:
        bundle: Structured manuscript handoff artifact.

    Returns:
        JSON-serializable representation of the bundle.
    """

    return to_llm_safe_jsonable(bundle)


def _hr(title: str = "") -> str:
    """Return a stable report divider."""
    width = 72
    if title:
        pad = width - len(title) - 2
        return f"\n-- {title} " + "-" * pad
    return "-" * width


def _format_entity_line(entity: CorpusEntity) -> str:
    """Render one canonical entity line for the manuscript report."""
    status = "REVIEW" if entity.review_required else "OK"
    docs = len(entity.supporting_document_paths)
    conflicts = (
        " conflicts=" + ",".join(category.value for category in entity.conflicting_categories)
        if entity.conflicting_categories else ""
    )
    aliases = (
        " keys=" + ",".join(entity.source_keys)
        if entity.source_keys != [entity.canonical_key] else ""
    )
    surfaces = (
        " absorbed_surfaces=" + ",".join(entity.absorbed_surface_forms)
        if entity.absorbed_surface_forms else ""
    )
    return (
        f"  {entity.canonical_key:20s}  {entity.dominant_category.value:10s}"
        f"  docs={docs:<2d}  conf={entity.aggregate_confidence:.3f}  {status}{conflicts}{aliases}{surfaces}"
    )


def _bucket_sort_key(bucket: DocumentEntityBucket) -> tuple[int, str]:
    """Return a stable presentation order for document entity buckets."""
    order = {
        DocumentEntityBucket.PROMOTED: 0,
        DocumentEntityBucket.REVIEW_ONLY: 1,
        DocumentEntityBucket.SUPPRESSED: 2,
    }
    return (order.get(bucket, 99), bucket.value)


def _category_sort_key(category: LexiconCategory) -> tuple[int, str]:
    """Return a stable presentation order for manuscript categories."""
    order = {
        LexiconCategory.CHARACTER: 0,
        LexiconCategory.GROUP: 1,
        LexiconCategory.PLACE: 2,
        LexiconCategory.OBJECT: 3,
        LexiconCategory.EVENT: 4,
        LexiconCategory.CONCEPT: 5,
        LexiconCategory.UNRESOLVED: 6,
    }
    return (order.get(category, 99), category.value)


def _preview_values(values: list[str], *, limit: int = 4) -> str:
    """Render a stable preview of a value list for the human report.

    Args:
        values: Values to render.
        limit: Maximum number of values to show before truncation.

    Returns:
        Compact preview string suitable for a report line.
    """
    if not values:
        return "-"
    shown = values[:limit]
    remainder = len(values) - len(shown)
    preview = ", ".join(shown)
    if remainder > 0:
        return f"{preview} (+{remainder} more)"
    return preview


def _format_category_trace(record) -> str:
    """Render the per-category score trace compactly for one record.

    Args:
        record: Document-local entity record.

    Returns:
        Stable score summary ordered by strongest category evidence first.
    """
    traces = sorted(
        record.classification_trace.evidence_by_category.values(),
        key=lambda item: (-item.score, item.category.value),
    )
    if not traces:
        return "-"
    return ", ".join(
        f"{trace.category.value}={trace.score:.3f}"
        for trace in traces
    )


def _format_runner_up(record) -> str:
    """Render the runner-up category summary for one record.

    Args:
        record: Document-local entity record.

    Returns:
        Stable runner-up rendering or a dash when absent.
    """
    runner_up = record.classification_trace.runner_up_category
    if runner_up is None:
        return "-"
    return f"{runner_up.value}:{record.classification_trace.runner_up_score:.3f}"


def _record_sort_key(record) -> tuple[str, int, str]:
    """Return a stable presentation order for document entity snapshots."""
    bucket_order = _bucket_sort_key(record.current_state.bucket)[0]
    return (
        record.identity.document_anchor.path,
        bucket_order,
        record.identity.normalized_key,
    )


def _render_record_snapshot(record) -> list[str]:
    """Render a compact readable snapshot of one document entity record.

    Args:
        record: Document-local entity record.

    Returns:
        Multi-line human-readable rendering of the record traces and profiles.
    """
    path = record.identity.document_anchor.path
    state = record.current_state
    source = record.source_evidence
    classification = record.classification_trace
    promotion = record.promotion_trace
    support = record.support_profile
    discourse = record.discourse_profile
    lineage = record.lineage_profile
    accepted = "yes" if classification.entityhood.accepted else "no"
    resolved = "yes" if state.resolved else "no"
    quote_only = "yes" if discourse.quote_only else "no"
    sentence_initial_only = "yes" if discourse.sentence_initial_only else "no"
    fully_covered = "yes" if lineage.fully_covered_by_longer_compound else "no"
    component = "yes" if lineage.appears_as_compound_component else "no"
    surface = "yes" if lineage.appears_as_compound_surface else "no"
    suppression = (
        promotion.suppression_reason.value
        if promotion.suppression_reason is not None else "-"
    )
    weaknesses = (
        ", ".join(classification.entityhood.weaknesses)
        if classification.entityhood.weaknesses else "-"
    )
    return [
        (
            f"  {record.identity.normalized_key:20s}  path={path}"
            f"  bucket={state.bucket.value}  cat={state.winning_category.value}"
            f"  resolved={resolved}"
        ),
        (
            "    identity: "
            f"record_id={record.identity.record_id}  "
            f"surfaces={_preview_values(record.identity.surface_forms)}"
        ),
        (
            "    source: "
            f"occ={source.occurrence_count:<2d}  anchors={len(source.anchors):<2d}"
            f"  windows={len(source.evidence_windows):<2d}"
            f"  attached_orbit={len(source.suppressed_related_evidence):<2d}"
        ),
        (
            "    class: "
            f"win={classification.winning_score:.3f}  "
            f"runner_up={_format_runner_up(record)}  "
            f"entityhood={classification.entityhood.score:.3f}  "
            f"accepted={accepted}"
        ),
        f"    entityhood_weaknesses: {weaknesses}",
        f"    class_scores: {_format_category_trace(record)}",
        (
            "    promo: "
            f"conf={promotion.confidence_score:.3f}  tier={promotion.rule_tier}"
            f"  scenes={promotion.scene_count}  attr={promotion.attribution_count}"
            f"  poss={promotion.possessive_count}  tfidf={promotion.tfidf_score:.3f}"
            f"  suppression={suppression}"
        ),
        (
            "    support: "
            f"title={support.title_support_count}  "
            f"poss={support.possessive_support_count}  "
            f"loc={support.location_support_count}  "
            f"linked_fields={support.linked_field_count}  "
            f"linked_definitions={support.linked_definition_count}  "
            f"linked_seeds={support.linked_seed_count}"
        ),
        (
            "    discourse: "
            f"quote={discourse.in_quote_count}  non_quote={discourse.non_quote_count}"
            f"  quote_only={quote_only}  sentence_initial={discourse.sentence_initial_count}"
            f"  sentence_initial_only={sentence_initial_only}"
            f"  address={discourse.address_like_count}"
            f"  attr_near={discourse.attributed_speaker_nearby_count}"
            f"  one_token={discourse.one_token_utterance_count}"
        ),
        (
            "    lineage: "
            f"parts={lineage.compound_part_count}  "
            f"fully_covered={fully_covered}  "
            f"covered={lineage.covered_anchor_count}  "
            f"uncovered={lineage.uncovered_anchor_count}  "
            f"component={component}  surface={surface}  "
            f"parents={_preview_values(lineage.candidate_parent_keys)}"
        ),
    ]


def render_manuscript_review_report(bundle: ManuscriptReviewBundle) -> str:
    """Render the human-readable manuscript handoff report from one bundle.

    Args:
        bundle: Persisted manuscript review artifact.

    Returns:
        Stable human-readable text report.
    """
    promoted_count = sum(
        1 for record in bundle.entity_records
        if record.current_state.bucket == DocumentEntityBucket.PROMOTED
    )
    review_count = sum(
        1 for record in bundle.entity_records
        if record.current_state.bucket == DocumentEntityBucket.REVIEW_ONLY
    )
    suppressed_count = sum(
        1 for record in bundle.entity_records
        if record.current_state.bucket == DocumentEntityBucket.SUPPRESSED
    )

    lines: list[str] = []
    lines.append(_hr("CORPUS"))
    lines.append(f"  Documents         : {len(bundle.document_paths)}")
    lines.append(f"  Entity records    : {len(bundle.entity_records)}")
    lines.append(f"  Canonical entities: {len(bundle.canonical_entities)}")
    lines.append(f"  Promoted records  : {promoted_count}")
    lines.append(f"  Review-only       : {review_count}")
    lines.append(f"  Suppressed        : {suppressed_count}")
    lines.append(
        "  Suppressed attached to entities: "
        f"{sum(len(record.source_evidence.suppressed_related_evidence) for record in bundle.entity_records if record.current_state.bucket != DocumentEntityBucket.SUPPRESSED)}"
    )

    lines.append(_hr("BUCKET SEMANTICS"))
    lines.append("  promoted    : foregrounded in the main entity inventory")
    lines.append("  review_only : preserved and visible for review, but not foregrounded")
    lines.append("  suppressed  : hidden from the main entity inventory, but retained")
    lines.append("                as semantic handoff evidence through local orbits")

    lines.append(_hr("FILES"))
    for path in bundle.document_paths:
        lines.append(f"  {path}")

    lines.append(_hr("DOCUMENT ENTITY RECORD SNAPSHOT"))
    lines.append(
        "  Selected examples expose the nested record contract in readable form."
    )
    lines.append(
        "  The full machine-readable record set remains in the JSON handoff artifact."
    )
    records_by_bucket: dict[DocumentEntityBucket, list[Any]] = {
        bucket: []
        for bucket in sorted(
            {record.current_state.bucket for record in bundle.entity_records},
            key=_bucket_sort_key,
        )
    }
    for record in sorted(bundle.entity_records, key=_record_sort_key):
        records_by_bucket.setdefault(record.current_state.bucket, []).append(record)
    for bucket in sorted(records_by_bucket, key=_bucket_sort_key):
        bucket_records = records_by_bucket[bucket]
        if not bucket_records:
            continue
        visible_records = [
            record for record in bucket_records
            if (
                record.discourse_profile.address_like_count > 0
                or record.discourse_profile.attributed_speaker_nearby_count > 0
                or record.lineage_profile.fully_covered_by_longer_compound
                or record.lineage_profile.covered_anchor_count > 0
                or record.support_profile.linked_field_count > 0
                or record.support_profile.linked_definition_count > 0
                or record.support_profile.linked_seed_count > 0
                or len(record.source_evidence.suppressed_related_evidence) > 0
            )
        ]
        shown_records = visible_records[:6]
        if len(shown_records) < min(6, len(bucket_records)):
            remaining = [record for record in bucket_records if record not in shown_records]
            shown_records.extend(remaining[: 6 - len(shown_records)])
        lines.append(
            f"  [{bucket.value}] total={len(bucket_records)} showing={len(shown_records)}"
        )
        for record in shown_records:
            lines.extend(_render_record_snapshot(record))

    sorted_entities = sorted(
        bundle.canonical_entities,
        key=lambda item: (
            -len(item.supporting_document_paths),
            -item.aggregate_confidence,
            item.canonical_key,
        ),
    )

    lines.append(_hr("CATEGORY COUNTS"))
    category_counts: dict[LexiconCategory, int] = {}
    for entity in sorted_entities:
        category_counts[entity.dominant_category] = (
            category_counts.get(entity.dominant_category, 0) + 1
        )
    for category in sorted(category_counts, key=_category_sort_key):
        lines.append(f"  {category.value:20s}  count={category_counts[category]}")

    lines.append(_hr("CANONICAL ENTITIES BY CATEGORY"))
    for category in sorted(category_counts, key=_category_sort_key):
        lines.append(f"  [{category.value}]")
        for entity in sorted_entities:
            if entity.dominant_category != category:
                continue
            lines.append(_format_entity_line(entity))

    lines.append(_hr("REVIEW REQUIRED"))
    review_entities = [entity for entity in bundle.canonical_entities if entity.review_required]
    if review_entities:
        for entity in sorted(review_entities, key=lambda item: item.canonical_key):
            lines.append(_format_entity_line(entity))
            for reason in entity.reasons:
                lines.append(f"    reason: {reason}")
    else:
        lines.append("  None.")

    lines.append(_hr("SEMANTIC REFERENCES"))
    reference_counts = {
        ReferenceCandidateType.BOUND_TITLE_ROLE: 0,
        ReferenceCandidateType.BARE_TITLE_ROLE: 0,
        ReferenceCandidateType.BOUND_RELATION_ROLE: 0,
        ReferenceCandidateType.BARE_RELATION_ROLE: 0,
    }
    for reference in bundle.reference_candidates:
        reference_counts[reference.reference_type] += 1
    lines.append(
        "  bound_title_role      mentions="
        f"{reference_counts[ReferenceCandidateType.BOUND_TITLE_ROLE]}"
    )
    lines.append(
        "  bare_title_role       mentions="
        f"{reference_counts[ReferenceCandidateType.BARE_TITLE_ROLE]}"
    )
    lines.append(
        "  bound_relation_role   mentions="
        f"{reference_counts[ReferenceCandidateType.BOUND_RELATION_ROLE]}"
    )
    lines.append(
        "  bare_relation_role    mentions="
        f"{reference_counts[ReferenceCandidateType.BARE_RELATION_ROLE]}"
    )
    lines.append(f"  grouped_reference_clusters  count={len(bundle.reference_clusters)}")
    for reference in sorted(
        bundle.reference_clusters,
        key=lambda item: (
            item.reference_type.value,
            item.document_anchor.path,
            -item.occurrence_count,
            item.normalized,
        ),
    )[:40]:
        links = ",".join(reference.candidate_entity_scores) if reference.candidate_entity_scores else "-"
        speakers = ",".join(reference.speaker_entity_scores) if reference.speaker_entity_scores else "-"
        suppressed = len(reference.suppressed_related_evidence)
        lines.append(
            f"  {reference.reference_type.value:20s}  {reference.normalized:12s}"
            f"  path={reference.document_anchor.path}  occ={reference.occurrence_count:<2d}"
            f"  addr={reference.address_like_count:<2d}  supp={suppressed:<2d}"
            f"  speakers={speakers}  links={links}"
        )

    lines.append(_hr("SUPPRESSED EVIDENCE ORBITS"))
    attached_records = [
        record for record in bundle.entity_records
        if record.current_state.bucket != DocumentEntityBucket.SUPPRESSED and record.source_evidence.suppressed_related_evidence
    ]
    if attached_records:
        for record in sorted(
            attached_records,
            key=lambda item: (
                item.identity.document_anchor.path,
                -len(item.source_evidence.suppressed_related_evidence),
                item.identity.normalized_key,
            ),
        )[:40]:
            lines.append(
                f"  {record.identity.normalized_key:20s}  path={record.identity.document_anchor.path}"
                f"  attached={len(record.source_evidence.suppressed_related_evidence)}"
            )
            for evidence in record.source_evidence.suppressed_related_evidence[:5]:
                lines.append(
                    f"    {evidence.normalized_key:20s}  {evidence.reason.value}"
                    f"  conf={evidence.confidence_score:.3f}"
                )
    else:
        lines.append("  None.")

    lines.append(_hr("SEMANTIC CONFLICTS"))
    if bundle.conflict_records:
        for conflict in sorted(bundle.conflict_records, key=lambda item: item.canonical_key):
            categories = ",".join(category.value for category in conflict.conflicting_categories)
            paths = ",".join(conflict.supporting_document_paths)
            lines.append(
                f"  {conflict.canonical_key:20s}  source={conflict.source.value}"
                f"  categories={categories}  paths={paths}"
            )
            lines.append(f"    reason: {conflict.reason}")
    else:
        lines.append("  None.")

    lines.append(_hr("CHARACTER SEMANTIC SUMMARIES"))
    if bundle.character_summaries:
        for summary in sorted(bundle.character_summaries, key=lambda item: item.canonical_key):
            lines.append(
                f"  {summary.canonical_key:20s}  docs={len(summary.supporting_document_paths):<2d}"
                f"  attribution={summary.aggregate_attribution_count}"
            )
            if summary.alias_keys:
                lines.append(f"    aliases: {', '.join(summary.alias_keys)}")
            if summary.canonical_surface_forms:
                lines.append(
                    f"    canonical_surfaces: {', '.join(summary.canonical_surface_forms)}"
                )
            if summary.absorbed_surface_forms:
                lines.append(
                    f"    absorbed_surfaces: {', '.join(summary.absorbed_surface_forms)}"
                )
            if summary.merge_reasons:
                lines.append(f"    merge_reasons: {'; '.join(summary.merge_reasons)}")
            if summary.attached_title_counts:
                lines.append(
                    "    attached_titles: "
                    + ", ".join(
                        f"{key}={value}"
                        for key, value in sorted(summary.attached_title_counts.items())
                    )
                )
            if summary.ambiguous_title_counts:
                lines.append(
                    "    ambiguous_titles: "
                    + ", ".join(
                        f"{key}={value}"
                        for key, value in sorted(summary.ambiguous_title_counts.items())
                    )
                )
            if summary.attached_relation_counts:
                lines.append(
                    "    attached_relations: "
                    + ", ".join(
                        f"{key}={value}"
                        for key, value in sorted(summary.attached_relation_counts.items())
                    )
                )
            if summary.ambiguous_relation_counts:
                lines.append(
                    "    ambiguous_relations: "
                    + ", ".join(
                        f"{key}={value}"
                        for key, value in sorted(summary.ambiguous_relation_counts.items())
                    )
                )
            if summary.conflict_sources:
                lines.append(
                    "    conflict_sources: "
                    + ", ".join(source.value for source in summary.conflict_sources)
                )
    else:
        lines.append("  None.")

    lines.append(_hr("SEMANTIC REVIEW TASKS"))
    lines.append("  Semantic handoff stops at review questions in this report.")
    if bundle.review_tasks:
        for task in sorted(bundle.review_tasks, key=lambda item: item.task_id):
            lines.append(f"  {task.kind.value:24s}  {task.subject_key}")
            lines.append(f"    {task.prompt}")
            lines.append(f"    paths: {', '.join(task.supporting_anchor_paths)}")
            if task.ranked_candidate_keys:
                lines.append(f"    ranked_candidates: {', '.join(task.ranked_candidate_keys)}")
            if task.ranked_speaker_keys:
                lines.append(f"    ranked_speakers: {', '.join(task.ranked_speaker_keys)}")
            if task.corpus_owner_keys:
                lines.append(f"    corpus_owners: {', '.join(task.corpus_owner_keys)}")
            if task.evidence_note:
                lines.append(f"    evidence_note: {task.evidence_note}")
    else:
        lines.append("  None.")

    lines.append(_hr("TOP DOCUMENT SUPPORT"))
    for entity in sorted(
        bundle.canonical_entities,
        key=lambda item: (
            -len(item.supporting_document_paths),
            -item.aggregate_confidence,
            item.canonical_key,
        ),
    )[:25]:
        lines.append(
            f"  {entity.canonical_key:20s}  paths="
            f"{', '.join(entity.supporting_document_paths)}"
        )

    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")
