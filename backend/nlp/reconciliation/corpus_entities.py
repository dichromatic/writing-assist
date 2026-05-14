"""
Cross-document entity reconciliation.

.. code-block:: mermaid

    flowchart TD
        A[DocumentEntityRecord list] --> B[Group by normalized_key]
        B --> C[Summarise document support]
        C --> D[Build exact-key CorpusEntity list]
        D --> E[Merge safe character aliases]
        E --> F[Merge safe non-character head aliases]
        F --> G[Merge safe non-character modifier aliases]
        G --> H[Merge safe contained non-character aliases]
        H --> I[Defer unresolved longer compounds to resolved anchors]
        I --> J[CorpusEntity list]
        I --> J[CorpusReconciliationResult]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.harvesting.shared import TITLE_PREFIXES_LOWER, has_generic_modifier_profile
from backend.nlp.types import (
    CorpusEntity,
    CorpusReconciliationResult,
    DocumentEntityBucket,
    DocumentEntityRecord,
    LexiconCategory,
)

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

    canonical_surface_forms = sorted({
        surface
        for record in members
        if record.normalized_key == canonical_key
        for surface in record.surface_forms
    })
    absorbed_surface_forms = sorted({
        surface
        for record in members
        if record.normalized_key != canonical_key
        for surface in record.surface_forms
    })

    return CorpusEntity(
        canonical_key=canonical_key,
        source_keys=sorted(source_keys),
        canonical_surface_forms=canonical_surface_forms,
        absorbed_surface_forms=absorbed_surface_forms,
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
            left in TITLE_PREFIXES_LOWER
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
            part in TITLE_PREFIXES_LOWER or has_generic_modifier_profile(part)
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


def _is_safe_non_character_head_component(
    component: CorpusEntity,
    compound_category: LexiconCategory,
    supporting_document_paths: list[str],
) -> bool:
    """Return True when a trailing head key can defer to a stronger compound.

    This pass is intentionally narrower than the character alias merger. It
    only handles resolved non-character compounds whose trailing head key looks
    like a shorter reference to the same entity in the same documents.

    Args:
        component: Corpus entity for the trailing head key.
        compound_category: Resolved category of the longer compound.
        supporting_document_paths: Documents where the longer compound appears.

    Returns:
        True when the trailing head key is category-compatible and does not
        require cross-document guessing outside the compound's support set.
    """
    if component.review_required and component.dominant_category != compound_category:
        return False
    if component.dominant_category not in {compound_category, LexiconCategory.UNRESOLVED}:
        return False
    return set(component.supporting_document_paths).issubset(set(supporting_document_paths))


def _merge_non_character_head_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer shorter head keys to stronger resolved non-character compounds.

    Compounds such as ``norre institute`` or ``polar north`` often surface as
    both a full name and a shorter trailing head reference. When the trailing
    head key only appears inside the same document set and points uniquely to
    one stronger resolved compound, the longer surface is the better canonical
    key for corpus reporting.

    Args:
        entities: Canonical entities after the character alias passes.
        all_records_by_key: All document-local records, including suppressed
            ones, keyed by exact normalized surface.

    Returns:
        Canonical entities with safe non-character head aliases merged into the
        stronger compound key.
    """
    by_key = {entity.canonical_key: entity for entity in entities}
    head_to_compounds: dict[str, set[str]] = defaultdict(set)
    eligible_compounds: dict[str, tuple[str, list[DocumentEntityRecord]]] = {}

    for key, anchor_records in all_records_by_key.items():
        parts = key.split()
        if len(parts) < 2:
            continue

        compound_entity = by_key.get(key)
        if compound_entity is None:
            continue
        if compound_entity.review_required:
            continue
        if compound_entity.dominant_category in {LexiconCategory.CHARACTER, LexiconCategory.UNRESOLVED}:
            continue

        head = parts[-1]
        head_entity = by_key.get(head)
        if head_entity is None:
            continue
        if key in head_entity.source_keys:
            continue

        anchor_paths = sorted({record.document_anchor.path for record in anchor_records})
        if not _is_safe_non_character_head_component(
            head_entity,
            compound_entity.dominant_category,
            anchor_paths,
        ):
            continue

        eligible_compounds[key] = (head, anchor_records)
        head_to_compounds[head].add(key)

    merge_targets: dict[str, list[tuple[str, list[DocumentEntityRecord], LexiconCategory]]] = defaultdict(list)
    absorbed_keys: set[str] = set()
    for key, (head, anchor_records) in sorted(eligible_compounds.items()):
        if head_to_compounds[head] != {key}:
            continue
        compound_entity = by_key[key]
        merge_targets[key].append((head, anchor_records, compound_entity.dominant_category))
        absorbed_keys.add(head)

    merged_entities: list[CorpusEntity] = []
    emitted_compounds: set[str] = set()
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        aliases = merge_targets.get(entity.canonical_key)
        if aliases:
            source_keys = set(entity.source_keys)
            combined_records = list(entity.member_records)
            for head, anchor_records, _category in aliases:
                source_keys.add(head)
                head_entity = by_key[head]
                source_keys.update(head_entity.source_keys)
                combined_records.extend(anchor_records)
                combined_records.extend(head_entity.member_records)
            combined_records = sorted(
                combined_records,
                key=lambda record: (record.document_anchor.path, record.normalized_key),
            )
            merged_entities.append(_build_corpus_entity(
                canonical_key=entity.canonical_key,
                source_keys=sorted(source_keys),
                members=combined_records,
                reasons=["resolved non-character compound absorbed its shorter head alias"],
            ))
            emitted_compounds.add(entity.canonical_key)
            continue

        if entity.canonical_key in absorbed_keys:
            continue

        if entity.canonical_key not in emitted_compounds:
            merged_entities.append(entity)

    return merged_entities


def _merge_non_character_modifier_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer shorter leading modifiers to stronger resolved compounds.

    Some corpus noise now comes from modifier-only keys such as ``radiant``
    surviving next to a stronger resolved compound like ``radiant estuary``.
    This pass only merges when the modifier key points uniquely to one resolved
    non-character compound of the same category within the same document set.

    Args:
        entities: Canonical entities after the head-alias merge pass.
        all_records_by_key: All document-local records, including suppressed
            ones, keyed by exact normalized surface.

    Returns:
        Canonical entities with safe modifier-only aliases folded into their
        stronger resolved compounds.
    """
    by_key = {entity.canonical_key: entity for entity in entities}
    modifier_to_compounds: dict[str, set[str]] = defaultdict(set)
    eligible_compounds: dict[str, tuple[str, list[DocumentEntityRecord]]] = {}

    for key, anchor_records in all_records_by_key.items():
        parts = key.split()
        if len(parts) < 2:
            continue

        compound_entity = by_key.get(key)
        if compound_entity is None:
            continue
        if compound_entity.review_required:
            continue
        if compound_entity.dominant_category in {LexiconCategory.CHARACTER, LexiconCategory.UNRESOLVED}:
            continue

        modifier = parts[0]
        modifier_entity = by_key.get(modifier)
        if modifier_entity is None:
            continue
        if key in modifier_entity.source_keys:
            continue
        if modifier_entity.dominant_category not in {
            compound_entity.dominant_category,
            LexiconCategory.UNRESOLVED,
        }:
            continue
        if modifier_entity.review_required and modifier_entity.dominant_category != compound_entity.dominant_category:
            continue

        anchor_paths = sorted({record.document_anchor.path for record in anchor_records})
        if not set(modifier_entity.supporting_document_paths).issubset(set(anchor_paths)):
            continue

        # Don't absorb a modifier that is mentioned more often than the compound.
        # If the shorter form is the more frequent reference, it is likely the
        # primary entity and the compound is a fuller variant of it, not the
        # other way around.
        modifier_occ = sum(r.occurrence_count for r in modifier_entity.member_records)
        compound_occ = sum(r.occurrence_count for r in by_key[key].member_records)
        if modifier_occ > compound_occ:
            continue

        eligible_compounds[key] = (modifier, anchor_records)
        modifier_to_compounds[modifier].add(key)

    merge_targets: dict[str, list[tuple[str, list[DocumentEntityRecord]]]] = defaultdict(list)
    absorbed_keys: set[str] = set()
    for key, (modifier, anchor_records) in sorted(eligible_compounds.items()):
        if modifier_to_compounds[modifier] != {key}:
            continue
        merge_targets[key].append((modifier, anchor_records))
        absorbed_keys.add(modifier)

    merged_entities: list[CorpusEntity] = []
    emitted_compounds: set[str] = set()
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        aliases = merge_targets.get(entity.canonical_key)
        if aliases:
            source_keys = set(entity.source_keys)
            combined_records = list(entity.member_records)
            for modifier, anchor_records in aliases:
                source_keys.add(modifier)
                modifier_entity = by_key[modifier]
                source_keys.update(modifier_entity.source_keys)
                combined_records.extend(anchor_records)
                combined_records.extend(modifier_entity.member_records)
            combined_records = sorted(
                combined_records,
                key=lambda record: (record.document_anchor.path, record.normalized_key),
            )
            merged_entities.append(_build_corpus_entity(
                canonical_key=entity.canonical_key,
                source_keys=sorted(source_keys),
                members=combined_records,
                reasons=["resolved non-character compound absorbed its shorter modifier alias"],
            ))
            emitted_compounds.add(entity.canonical_key)
            continue

        if entity.canonical_key in absorbed_keys:
            continue

        if entity.canonical_key not in emitted_compounds:
            merged_entities.append(entity)

    return merged_entities


def _merge_non_character_contained_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer shorter contained compounds to stronger resolved compounds.

    Some remaining corpus noise comes from longer resolved compounds and their
    shorter contained aliases surviving side by side, for example
    ``amerhinn remembrance gardens`` next to ``remembrance gardens``. This
    pass only merges contained aliases when the shorter phrase is multi-token,
    category-compatible, document-subset compatible, and uniquely owned by one
    stronger resolved compound.

    Args:
        entities: Canonical entities after the simpler alias passes.
        all_records_by_key: All document-local records, including suppressed
            ones, keyed by exact normalized surface.

    Returns:
        Canonical entities with safe contained multi-token aliases folded into
        their stronger resolved compounds.
    """
    by_key = {entity.canonical_key: entity for entity in entities}
    alias_to_compounds: dict[str, set[str]] = defaultdict(set)
    eligible_compounds: dict[str, list[tuple[str, list[DocumentEntityRecord]]]] = defaultdict(list)

    for key, anchor_records in all_records_by_key.items():
        parts = key.split()
        if len(parts) < 3:
            continue

        compound_entity = by_key.get(key)
        if compound_entity is None:
            continue
        if compound_entity.review_required:
            continue
        if compound_entity.dominant_category in {LexiconCategory.CHARACTER, LexiconCategory.UNRESOLVED}:
            continue

        anchor_paths = sorted({record.document_anchor.path for record in anchor_records})
        # Only consider suffix aliases (remove leading modifier word, e.g. "remembrance
        # gardens" from "amerhinn remembrance gardens"). Prefix aliases (remove trailing
        # qualifier, e.g. "east lagoon" from "east lagoon villa") have the wrong
        # absorption direction: the shorter form is the primary place name and the
        # trailing word is a type qualifier, not a geographic modifier.
        candidate_aliases = {" ".join(parts[1:])}
        for alias_key in candidate_aliases:
            alias_parts = alias_key.split()
            if len(alias_parts) < 2:
                continue

            alias_entity = by_key.get(alias_key)
            if alias_entity is None:
                continue
            if key in alias_entity.source_keys:
                continue
            if alias_entity.dominant_category not in {
                compound_entity.dominant_category,
                LexiconCategory.UNRESOLVED,
            }:
                continue
            if alias_entity.review_required and alias_entity.dominant_category != compound_entity.dominant_category:
                continue
            if not set(alias_entity.supporting_document_paths).issubset(set(anchor_paths)):
                continue

            # Don't absorb an alias that is mentioned more often than the compound.
            # If the shorter phrase is the more frequent reference, it is the
            # primary entity and the longer compound is a variant, not canonical.
            alias_occ = sum(r.occurrence_count for r in alias_entity.member_records)
            compound_occ = sum(r.occurrence_count for r in compound_entity.member_records)
            if alias_occ > compound_occ:
                continue

            eligible_compounds[key].append((alias_key, anchor_records))
            alias_to_compounds[alias_key].add(key)

    merge_targets: dict[str, list[tuple[str, list[DocumentEntityRecord]]]] = defaultdict(list)
    absorbed_keys: set[str] = set()
    for key, alias_entries in sorted(eligible_compounds.items()):
        for alias_key, anchor_records in alias_entries:
            if alias_to_compounds[alias_key] != {key}:
                continue
            merge_targets[key].append((alias_key, anchor_records))
            absorbed_keys.add(alias_key)

    merged_entities: list[CorpusEntity] = []
    emitted_compounds: set[str] = set()
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        aliases = merge_targets.get(entity.canonical_key)
        if aliases:
            source_keys = set(entity.source_keys)
            combined_records = list(entity.member_records)
            for alias_key, anchor_records in aliases:
                source_keys.add(alias_key)
                alias_entity = by_key[alias_key]
                source_keys.update(alias_entity.source_keys)
                combined_records.extend(anchor_records)
                combined_records.extend(alias_entity.member_records)
            combined_records = sorted(
                combined_records,
                key=lambda record: (record.document_anchor.path, record.normalized_key),
            )
            merged_entities.append(_build_corpus_entity(
                canonical_key=entity.canonical_key,
                source_keys=sorted(source_keys),
                members=combined_records,
                reasons=["resolved non-character compound absorbed its shorter contained alias"],
            ))
            emitted_compounds.add(entity.canonical_key)
            continue

        if entity.canonical_key in absorbed_keys:
            continue

        if entity.canonical_key not in emitted_compounds:
            merged_entities.append(entity)

    return merged_entities


def _defer_unresolved_longer_compounds_to_resolved_anchors(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer weak longer unresolved compounds to stronger resolved anchors.

    Some longer compounds still survive as unresolved canonicals even though a
    shorter contained alias already resolves cleanly elsewhere in the corpus.
    This pass only folds those longer unresolved phrases into one uniquely
    matched resolved non-character anchor when the relationship is already
    visible through exact surfaces or absorbed source keys.

    Args:
        entities: Canonical entities after the resolved non-character passes.
        all_records_by_key: All document-local records, including suppressed
            ones, keyed by exact normalized surface.

    Returns:
        Canonical entities with safe longer unresolved compounds folded into
        stronger resolved non-character anchors.
    """
    by_key = {entity.canonical_key: entity for entity in entities}
    source_key_index: dict[str, list[CorpusEntity]] = defaultdict(list)
    for entity in entities:
        for key in entity.source_keys:
            source_key_index[key].append(entity)

    def _resolved_targets(alias_key: str) -> list[CorpusEntity]:
        return [
            entity for entity in source_key_index.get(alias_key, [])
            if not entity.review_required
            and entity.dominant_category not in {LexiconCategory.CHARACTER, LexiconCategory.UNRESOLVED}
        ]

    defer_targets: dict[str, str] = {}
    deferred_anchor_records: dict[str, list[DocumentEntityRecord]] = {}
    deferred_children: set[str] = set()
    target_to_children: dict[str, list[str]] = defaultdict(list)

    for entity in entities:
        if entity.dominant_category != LexiconCategory.UNRESOLVED:
            continue

        parts = entity.canonical_key.split()
        if len(parts) < 3:
            continue

        alias_candidates = {
            " ".join(parts[1:]),
            " ".join(parts[:-1]),
        }
        resolved_targets = {
            target.canonical_key: target
            for alias_key in alias_candidates
            for target in _resolved_targets(alias_key)
            if set(entity.supporting_document_paths).issubset(set(target.supporting_document_paths))
        }
        if len(resolved_targets) != 1:
            continue

        target_key = next(iter(resolved_targets))
        defer_targets[entity.canonical_key] = target_key
        deferred_anchor_records[entity.canonical_key] = all_records_by_key.get(entity.canonical_key, entity.member_records)
        deferred_children.add(entity.canonical_key)
        target_to_children[target_key].append(entity.canonical_key)

    merged_entities: list[CorpusEntity] = []
    emitted_targets: set[str] = set()
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        child_keys = target_to_children.get(entity.canonical_key)
        if child_keys:
            source_keys = set(entity.source_keys)
            combined_records = list(entity.member_records)
            for child_key in child_keys:
                child_entity = by_key[child_key]
                source_keys.update(child_entity.source_keys)
                source_keys.add(child_key)
                combined_records.extend(child_entity.member_records)
                combined_records.extend(deferred_anchor_records[child_key])
            combined_records = sorted(
                combined_records,
                key=lambda record: (record.document_anchor.path, record.normalized_key),
            )
            merged_entities.append(_build_corpus_entity(
                canonical_key=entity.canonical_key,
                source_keys=sorted(source_keys),
                members=combined_records,
                reasons=["longer unresolved compound deferred to stronger resolved non-character anchor"],
            ))
            emitted_targets.add(entity.canonical_key)
            continue

        if entity.canonical_key in deferred_children:
            continue

        if entity.canonical_key not in emitted_targets:
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
    canonical_entities = _merge_non_character_head_aliases(canonical_entities, all_grouped)
    canonical_entities = _merge_non_character_modifier_aliases(canonical_entities, all_grouped)
    canonical_entities = _merge_non_character_contained_aliases(canonical_entities, all_grouped)
    canonical_entities = _defer_unresolved_longer_compounds_to_resolved_anchors(canonical_entities, all_grouped)
    return CorpusReconciliationResult(
        canonical_entities=sorted(canonical_entities, key=lambda entity: entity.canonical_key)
    )
