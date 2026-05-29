# Diagram omitted - utility module with no significant information flow.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from backend.nlp.types import CorpusEntity, DocumentEntityRecord, LexiconCategory


@dataclass(frozen=True)
class MergeChild:
    """One absorbed child alias plus any pass-specific anchor records.

    Args:
        child_key: Canonical key of the absorbed child entity.
        anchor_records: Extra direct records that should be retained when the
            child is folded into its stronger parent canonical.
    """

    child_key: str
    anchor_records: list[DocumentEntityRecord]


def dominant_category(records: list[DocumentEntityRecord]) -> tuple[LexiconCategory, list[LexiconCategory]]:
    """Choose the strongest current category for an exact-key group.

    Resolved non-unresolved categories take precedence. Unresolved-only groups
    stay unresolved until later phases add aliasing and richer corpus evidence.
    """
    resolved_records = [
        record for record in records
        if record.current_state.resolved and record.current_state.winning_category != LexiconCategory.UNRESOLVED
    ]
    if not resolved_records:
        return LexiconCategory.UNRESOLVED, []

    by_category: dict[LexiconCategory, list[DocumentEntityRecord]] = defaultdict(list)
    for record in resolved_records:
        by_category[record.current_state.winning_category].append(record)

    conflicting = sorted(by_category.keys(), key=lambda category: category.value)
    dominant = max(
        by_category.items(),
        key=lambda item: (
            len(item[1]),
            max(record.promotion_trace.confidence_score for record in item[1]),
            item[0].value,
        ),
    )[0]
    return dominant, conflicting


def build_corpus_entity(
    canonical_key: str,
    source_keys: list[str],
    members: list[DocumentEntityRecord],
    reasons: list[str],
) -> CorpusEntity:
    """Construct a stable corpus entity summary from grouped member records.

    Args:
        canonical_key: Canonical key to expose for the merged entity.
        source_keys: Exact normalized keys absorbed into this canonical entity.
        members: Document-local member records supporting the entity.
        reasons: Human-readable merge rationale.

    Returns:
        CorpusEntity ready for report output and later refinement.
    """
    chosen_category, conflicting_categories = dominant_category(members)
    review_required = len(conflicting_categories) > 1
    final_reasons = list(reasons)
    if review_required:
        final_reasons.append("conflicting resolved categories across documents")
    elif chosen_category == LexiconCategory.UNRESOLVED:
        final_reasons.append("no resolved category agreement yet across documents")

    canonical_surface_forms = sorted({
        surface
        for record in members
        if record.identity.normalized_key == canonical_key
        for surface in record.identity.surface_forms
    })
    absorbed_surface_forms = sorted({
        surface
        for record in members
        if record.identity.normalized_key != canonical_key
        for surface in record.identity.surface_forms
    })

    return CorpusEntity(
        canonical_key=canonical_key,
        source_keys=sorted(source_keys),
        canonical_surface_forms=canonical_surface_forms,
        absorbed_surface_forms=absorbed_surface_forms,
        member_records=members,
        supporting_document_paths=sorted({record.identity.document_anchor.path for record in members}),
        dominant_category=chosen_category,
        aggregate_confidence=max(record.promotion_trace.confidence_score for record in members),
        conflicting_categories=conflicting_categories if review_required else [],
        review_required=review_required,
        reasons=final_reasons,
    )


def apply_merge_plan(
    entities: list[CorpusEntity],
    by_key: dict[str, CorpusEntity],
    merge_targets: dict[str, list[MergeChild]],
    absorbed_keys: set[str],
    reason: str,
) -> list[CorpusEntity]:
    """Apply one precomputed alias merge plan and rebuild merged entities.

    Each alias pass decides eligibility differently, but once the parent-child
    pairs are fixed they all rebuild the merged canonical entity the same way.
    This helper preserves the current behavior: sorted entity iteration,
    sorted combined member records, absorbed-key filtering, and the exact
    user-visible reason string supplied by the caller.

    Args:
        entities: Canonical entities entering the merge pass.
        by_key: Fast lookup from canonical key to corpus entity.
        merge_targets: Parent canonical key to absorbed child plan entries.
        absorbed_keys: Canonical keys that should disappear after merging.
        reason: Exact merge reason string to emit on rebuilt entities.

    Returns:
        Canonical entities after applying the merge plan.
    """
    merged_entities: list[CorpusEntity] = []
    emitted_parents: set[str] = set()

    for entity in sorted(entities, key=lambda item: item.canonical_key):
        children = merge_targets.get(entity.canonical_key)
        if children:
            source_keys = set(entity.source_keys)
            combined_records = list(entity.member_records)
            for child in children:
                child_entity = by_key[child.child_key]
                source_keys.add(child.child_key)
                source_keys.update(child_entity.source_keys)
                combined_records.extend(child.anchor_records)
                combined_records.extend(child_entity.member_records)
            combined_records = sorted(
                combined_records,
                key=lambda record: (record.identity.document_anchor.path, record.identity.normalized_key),
            )
            merged_entities.append(build_corpus_entity(
                canonical_key=entity.canonical_key,
                source_keys=sorted(source_keys),
                members=combined_records,
                reasons=[reason],
            ))
            emitted_parents.add(entity.canonical_key)
            continue

        if entity.canonical_key in absorbed_keys:
            continue

        if entity.canonical_key not in emitted_parents:
            merged_entities.append(entity)

    return merged_entities
