"""
Cross-document entity reconciliation.

.. code-block:: mermaid

    flowchart TD
        A[DocumentEntityRecord list] --> B[Group by normalized_key]
        B --> C[Summarise document support]
        C --> D[Build exact-key CorpusEntity list]
        D --> E[Merge safe character aliases]
        E --> F[CorpusEntity list]
        E --> F[CorpusReconciliationResult]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.harvesting.shared import TITLE_PREFIXES, has_generic_modifier_profile
from backend.nlp.types import (
    CorpusEntity,
    CorpusReconciliationResult,
    DocumentEntityBucket,
    DocumentEntityRecord,
    LexiconCategory,
)

_TITLE_PREFIX_NORMALIZED = frozenset(title.lower() for title in TITLE_PREFIXES)


def _dominant_category(records: list[DocumentEntityRecord]) -> tuple[LexiconCategory, list[LexiconCategory]]:
    """Choose the strongest current category for an exact-key group.

    Resolved non-unresolved categories take precedence. Unresolved-only groups
    stay unresolved until later phases add aliasing and richer corpus evidence.
    """
    resolved_records = [
        record for record in records
        if record.resolved and record.winning_category != LexiconCategory.UNRESOLVED
    ]
    if not resolved_records:
        return LexiconCategory.UNRESOLVED, []

    by_category: dict[LexiconCategory, list[DocumentEntityRecord]] = defaultdict(list)
    for record in resolved_records:
        by_category[record.winning_category].append(record)

    conflicting = sorted(by_category.keys(), key=lambda category: category.value)
    dominant = max(
        by_category.items(),
        key=lambda item: (
            len(item[1]),
            max(record.confidence_score for record in item[1]),
            item[0].value,
        ),
    )[0]
    return dominant, conflicting


def _build_corpus_entity(
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
    dominant_category, conflicting_categories = _dominant_category(members)
    review_required = len(conflicting_categories) > 1
    final_reasons = list(reasons)
    if review_required:
        final_reasons.append("conflicting resolved categories across documents")
    elif dominant_category == LexiconCategory.UNRESOLVED:
        final_reasons.append("no resolved category agreement yet across documents")

    return CorpusEntity(
        canonical_key=canonical_key,
        source_keys=sorted(source_keys),
        member_records=members,
        supporting_document_paths=sorted({record.document_anchor.path for record in members}),
        dominant_category=dominant_category,
        aggregate_confidence=max(record.confidence_score for record in members),
        conflicting_categories=conflicting_categories if review_required else [],
        review_required=review_required,
        reasons=final_reasons,
    )


def _is_safe_character_component(
    component: CorpusEntity,
    supporting_document_paths: list[str],
) -> bool:
    """Return True when a single-token entity can fold into a character compound.

    The first alias phase is intentionally narrow. Components are only folded
    into a compound character when they already look person-like themselves
    and share at least one supporting document with the compound. This keeps
    place/event compounds out of character canonicalization.
    """
    if component.review_required and component.dominant_category != LexiconCategory.CHARACTER:
        return False
    if component.dominant_category not in {LexiconCategory.CHARACTER, LexiconCategory.UNRESOLVED}:
        return False
    return bool(
        set(component.supporting_document_paths)
        & set(supporting_document_paths)
    )


def _merge_character_compound_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Merge unambiguous character full-name compounds over exact-key entities.

    A compound such as ``tsushima yoshiko`` is a stronger canonical identity
    surface than its component keys when the corpus contains the full form and
    the single-token parts also behave like the same character. This pass only
    merges character compounds with exactly two parts and only when both parts
    uniquely point back to the same compound candidate.
    """
    by_key = {entity.canonical_key: entity for entity in entities}
    eligible_compounds: dict[str, tuple[tuple[str, str], list[DocumentEntityRecord]]] = {}
    component_to_compounds: dict[str, set[str]] = defaultdict(set)

    for key, anchor_records in all_records_by_key.items():
        parts = key.split()
        if len(parts) != 2:
            continue

        left, right = parts
        left_entity = by_key.get(left)
        right_entity = by_key.get(right)
        if left_entity is None or right_entity is None:
            continue

        anchor_paths = sorted({record.document_anchor.path for record in anchor_records})
        if not _is_safe_character_component(left_entity, anchor_paths):
            continue
        if not _is_safe_character_component(right_entity, anchor_paths):
            continue

        anchor_entity = by_key.get(key)
        if anchor_entity is not None and (
            anchor_entity.review_required
            or anchor_entity.dominant_category not in {LexiconCategory.CHARACTER, LexiconCategory.UNRESOLVED}
        ):
            continue

        eligible_compounds[key] = ((left, right), anchor_records)
        component_to_compounds[left].add(key)
        component_to_compounds[right].add(key)

    merge_plans: dict[str, tuple[str, str]] = {}
    merge_canonical_keys: dict[str, str] = {}
    merge_anchor_records: dict[str, list[DocumentEntityRecord]] = {}
    merged_children: set[str] = set()

    for key, (parts, anchor_records) in sorted(eligible_compounds.items()):
        left, right = parts
        if component_to_compounds[left] != {key}:
            continue
        if component_to_compounds[right] != {key}:
            continue
        merge_plans[key] = parts
        if (
            left in _TITLE_PREFIX_NORMALIZED
            or (
                by_key[left].dominant_category != LexiconCategory.CHARACTER
                and by_key[right].dominant_category == LexiconCategory.CHARACTER
            )
        ):
            merge_canonical_keys[key] = right
        elif (
            by_key[right].dominant_category != LexiconCategory.CHARACTER
            and by_key[left].dominant_category == LexiconCategory.CHARACTER
        ):
            merge_canonical_keys[key] = left
        else:
            merge_canonical_keys[key] = key
        merge_anchor_records[key] = anchor_records
        merged_children.update({left, right, key})

    emitted_canonicals: set[str] = set()
    merged_entities: list[CorpusEntity] = []
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        parts = merge_plans.get(entity.canonical_key)
        if parts is not None:
            left, right = parts
            canonical_key = merge_canonical_keys[entity.canonical_key]
            left_entity = by_key[left]
            right_entity = by_key[right]
            if canonical_key not in emitted_canonicals:
                combined_records = sorted(
                    merge_anchor_records[entity.canonical_key] + left_entity.member_records + right_entity.member_records,
                    key=lambda record: (record.document_anchor.path, record.normalized_key),
                )
                reasons = ["character compound merged with its single-token alias components"]
                if canonical_key != entity.canonical_key:
                    reasons.append("titled or role-led compound deferred to stronger personal key")
                merged_entities.append(_build_corpus_entity(
                    canonical_key=canonical_key,
                    source_keys=[entity.canonical_key, left, right],
                    members=combined_records,
                    reasons=reasons,
                ))
                emitted_canonicals.add(canonical_key)
            continue

        if entity.canonical_key in merged_children:
            continue

        merged_entities.append(entity)

    for key in sorted(set(merge_plans) - {entity.canonical_key for entity in entities}):
        left, right = merge_plans[key]
        canonical_key = merge_canonical_keys[key]
        left_entity = by_key[left]
        right_entity = by_key[right]
        if canonical_key in emitted_canonicals:
            continue
        combined_records = sorted(
            merge_anchor_records[key] + left_entity.member_records + right_entity.member_records,
            key=lambda record: (record.document_anchor.path, record.normalized_key),
        )
        reasons = ["sparse observed character compound merged with its single-token alias components"]
        if canonical_key != key:
            reasons.append("titled or role-led compound deferred to stronger personal key")
        merged_entities.append(_build_corpus_entity(
            canonical_key=canonical_key,
            source_keys=[key, left, right],
            members=combined_records,
            reasons=reasons,
        ))
        emitted_canonicals.add(canonical_key)

    return merged_entities


def _merge_generic_leading_character_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer generic-leading character compounds to the trailing personal key.

    Phrases such as ``old man hiroshi`` behave more like decorated aliases than
    independent canonicals when the final token already has stronger character
    evidence. This pass keeps the full surface in source_keys while exposing the
    cleaner personal key as the corpus canonical.
    """
    by_key = {entity.canonical_key: entity for entity in entities}
    merge_targets: dict[str, list[tuple[str, list[DocumentEntityRecord]]]] = defaultdict(list)
    absorbed_keys: set[str] = set()

    for key, anchor_records in all_records_by_key.items():
        parts = key.split()
        if len(parts) < 2:
            continue

        tail = parts[-1]
        tail_entity = by_key.get(tail)
        if tail_entity is None:
            continue
        if key in tail_entity.source_keys:
            continue

        anchor_paths = sorted({record.document_anchor.path for record in anchor_records})
        if not _is_safe_character_component(tail_entity, anchor_paths):
            continue

        if not all(
            part in _TITLE_PREFIX_NORMALIZED or has_generic_modifier_profile(part)
            for part in parts[:-1]
        ):
            continue

        anchor_entity = by_key.get(key)
        if anchor_entity is not None and (
            anchor_entity.review_required
            or anchor_entity.dominant_category not in {LexiconCategory.CHARACTER, LexiconCategory.UNRESOLVED}
        ):
            continue

        merge_targets[tail].append((key, anchor_records))
        absorbed_keys.add(key)

    merged_entities: list[CorpusEntity] = []
    emitted_canonicals: set[str] = set()
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        aliases = merge_targets.get(entity.canonical_key)
        if aliases:
            source_keys = set(entity.source_keys)
            combined_records = list(entity.member_records)
            for key, anchor_records in aliases:
                source_keys.add(key)
                combined_records.extend(anchor_records)
            combined_records = sorted(
                combined_records,
                key=lambda record: (record.document_anchor.path, record.normalized_key),
            )
            merged_entities.append(_build_corpus_entity(
                canonical_key=entity.canonical_key,
                source_keys=sorted(source_keys),
                members=combined_records,
                reasons=["generic-leading character compounds deferred to stronger personal key"],
            ))
            emitted_canonicals.add(entity.canonical_key)
            continue

        if entity.canonical_key in absorbed_keys:
            continue

        if entity.canonical_key not in emitted_canonicals:
            merged_entities.append(entity)

    return merged_entities


def reconcile_document_entities(
    records: list[DocumentEntityRecord],
    *,
    include_suppressed: bool = False,
) -> CorpusReconciliationResult:
    """Merge document-local entity records into corpus-level canonical entities.

    The reconciliation stage still starts from exact normalized keys, preserves
    per-document decisions, and surfaces category conflicts explicitly. It now
    also adds a narrow character-alias pass so full-name compounds can absorb
    their single-token character components when that merge is unambiguous.

    Args:
        records: Per-document entity summaries from summarize_document_entities.
        include_suppressed: When True, also include suppressed document-local
            records. The default excludes them so canonical corpus entities are
            built from promoted and review-only evidence rather than noise.

    Returns:
        CorpusReconciliationResult with canonical entities ready for later
        fuzzy alias and cross-document refinement.
    """
    grouped: dict[str, list[DocumentEntityRecord]] = defaultdict(list)
    all_grouped: dict[str, list[DocumentEntityRecord]] = defaultdict(list)
    for record in records:
        all_grouped[record.normalized_key].append(record)
        if not include_suppressed and record.bucket == DocumentEntityBucket.SUPPRESSED:
            continue
        grouped[record.normalized_key].append(record)

    exact_entities: list[CorpusEntity] = []
    for key in sorted(grouped):
        members = sorted(grouped[key], key=lambda record: (record.document_anchor.path, record.normalized_key))
        exact_entities.append(_build_corpus_entity(
            canonical_key=key,
            source_keys=[key],
            members=members,
            reasons=["exact normalized key matched across documents"],
        ))

    canonical_entities = _merge_character_compound_aliases(exact_entities, all_grouped)
    canonical_entities = _merge_generic_leading_character_aliases(canonical_entities, all_grouped)
    return CorpusReconciliationResult(
        canonical_entities=sorted(canonical_entities, key=lambda entity: entity.canonical_key)
    )
