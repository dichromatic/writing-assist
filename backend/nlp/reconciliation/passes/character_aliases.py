"""
Character-oriented reconciliation passes.

.. code-block:: mermaid

    flowchart TD
        A[CorpusEntity list + all exact-key records] --> B[Find eligible character compounds]
        B --> C[Resolve canonical direction]
        C --> D[Emit merged character canonicals]
        A --> E[Find generic-leading character aliases]
        E --> F[Defer to stronger personal key]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from backend.nlp.harvesting.shared import TITLE_PREFIXES_LOWER, has_generic_modifier_profile
from backend.nlp.reconciliation.passes.common import build_corpus_entity
from backend.nlp.types import CorpusEntity, DocumentEntityRecord, LexiconCategory


@dataclass(frozen=True)
class _EligibleCharacterCompound:
    """Prevalidated two-token character compound candidate.

    Args:
        compound_key: Observed two-token compound surface.
        left_key: Left component key of the compound.
        right_key: Right component key of the compound.
        anchor_records: All exact records observed for the compound surface.
    """

    compound_key: str
    left_key: str
    right_key: str
    anchor_records: list[DocumentEntityRecord]


@dataclass(frozen=True)
class _CharacterCompoundMergePlan:
    """Resolved merge plan for one character-compound canonicalization.

    Args:
        source_key: Observed compound key that triggered this merge.
        canonical_key: Final canonical key to emit for the merged entity.
        left_key: Left component that will be absorbed.
        right_key: Right component that will be absorbed.
        anchor_records: Exact compound records to keep as supporting evidence.
        sparse_observed: True when the compound exists only as raw records and
            not as an input corpus entity.
    """

    source_key: str
    canonical_key: str
    left_key: str
    right_key: str
    anchor_records: list[DocumentEntityRecord]
    sparse_observed: bool


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
    return bool(set(component.supporting_document_paths) & set(supporting_document_paths))


def _identify_eligible_character_compounds(
    by_key: dict[str, CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> tuple[dict[str, _EligibleCharacterCompound], dict[str, set[str]]]:
    """Return character compounds whose components safely support merging."""
    eligible_compounds: dict[str, _EligibleCharacterCompound] = {}
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

        eligible_compounds[key] = _EligibleCharacterCompound(
            compound_key=key,
            left_key=left,
            right_key=right,
            anchor_records=anchor_records,
        )
        component_to_compounds[left].add(key)
        component_to_compounds[right].add(key)

    return eligible_compounds, component_to_compounds


def _build_character_merge_plans(
    by_key: dict[str, CorpusEntity],
    eligible_compounds: dict[str, _EligibleCharacterCompound],
    component_to_compounds: dict[str, set[str]],
    title_prefixes_lower: frozenset[str],
) -> dict[str, _CharacterCompoundMergePlan]:
    """Resolve canonical direction for uniquely owned character compounds."""
    merge_plans: dict[str, _CharacterCompoundMergePlan] = {}
    for key, candidate in sorted(eligible_compounds.items()):
        left = candidate.left_key
        right = candidate.right_key
        if component_to_compounds[left] != {key}:
            continue
        if component_to_compounds[right] != {key}:
            continue

        if (
            left in title_prefixes_lower
            or (
                by_key[left].dominant_category != LexiconCategory.CHARACTER
                and by_key[right].dominant_category == LexiconCategory.CHARACTER
            )
        ):
            canonical_key = right
        elif (
            by_key[right].dominant_category != LexiconCategory.CHARACTER
            and by_key[left].dominant_category == LexiconCategory.CHARACTER
        ):
            canonical_key = left
        else:
            canonical_key = key

        merge_plans[key] = _CharacterCompoundMergePlan(
            source_key=key,
            canonical_key=canonical_key,
            left_key=left,
            right_key=right,
            anchor_records=candidate.anchor_records,
            sparse_observed=key not in by_key,
        )
    return merge_plans


def _emit_character_compound_merges(
    entities: list[CorpusEntity],
    by_key: dict[str, CorpusEntity],
    merge_plans: dict[str, _CharacterCompoundMergePlan],
) -> list[CorpusEntity]:
    """Emit merged corpus entities from precomputed character merge plans."""
    merged_children: set[str] = set()
    for plan in merge_plans.values():
        merged_children.update({plan.source_key, plan.left_key, plan.right_key})

    emitted_canonicals: set[str] = set()
    merged_entities: list[CorpusEntity] = []
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        plan = merge_plans.get(entity.canonical_key)
        if plan is not None:
            left_entity = by_key[plan.left_key]
            right_entity = by_key[plan.right_key]
            if plan.canonical_key not in emitted_canonicals:
                combined_records = sorted(
                    plan.anchor_records + left_entity.member_records + right_entity.member_records,
                    key=lambda record: (record.document_anchor.path, record.normalized_key),
                )
                reasons = ["character compound merged with its single-token alias components"]
                if plan.canonical_key != plan.source_key:
                    reasons.append("titled or role-led compound deferred to stronger personal key")
                merged_entities.append(build_corpus_entity(
                    canonical_key=plan.canonical_key,
                    source_keys=[plan.source_key, plan.left_key, plan.right_key],
                    members=combined_records,
                    reasons=reasons,
                ))
                emitted_canonicals.add(plan.canonical_key)
            continue

        if entity.canonical_key in merged_children:
            continue

        merged_entities.append(entity)

    for key in sorted(set(merge_plans) - {entity.canonical_key for entity in entities}):
        plan = merge_plans[key]
        if plan.canonical_key in emitted_canonicals:
            continue
        left_entity = by_key[plan.left_key]
        right_entity = by_key[plan.right_key]
        combined_records = sorted(
            plan.anchor_records + left_entity.member_records + right_entity.member_records,
            key=lambda record: (record.document_anchor.path, record.normalized_key),
        )
        reasons = ["sparse observed character compound merged with its single-token alias components"]
        if plan.canonical_key != plan.source_key:
            reasons.append("titled or role-led compound deferred to stronger personal key")
        merged_entities.append(build_corpus_entity(
            canonical_key=plan.canonical_key,
            source_keys=[plan.source_key, plan.left_key, plan.right_key],
            members=combined_records,
            reasons=reasons,
        ))
        emitted_canonicals.add(plan.canonical_key)

    return merged_entities


def merge_character_compound_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
    title_prefixes_lower: frozenset[str] = TITLE_PREFIXES_LOWER,
) -> list[CorpusEntity]:
    """Merge unambiguous character full-name compounds over exact-key entities."""
    by_key = {entity.canonical_key: entity for entity in entities}
    eligible_compounds, component_to_compounds = _identify_eligible_character_compounds(
        by_key,
        all_records_by_key,
    )
    merge_plans = _build_character_merge_plans(
        by_key,
        eligible_compounds,
        component_to_compounds,
        title_prefixes_lower,
    )
    return _emit_character_compound_merges(entities, by_key, merge_plans)


def merge_generic_leading_character_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
    title_prefixes_lower: frozenset[str] = TITLE_PREFIXES_LOWER,
) -> list[CorpusEntity]:
    """Defer generic-leading character compounds to the trailing personal key."""
    by_key = {entity.canonical_key: entity for entity in entities}
    source_to_canonical: dict[str, str] = {
        src: entity.canonical_key
        for entity in entities
        for src in entity.source_keys
        if src != entity.canonical_key
    }
    merge_targets: dict[str, list[tuple[str, list[DocumentEntityRecord]]]] = defaultdict(list)
    absorbed_keys: set[str] = set()

    for key, anchor_records in all_records_by_key.items():
        parts = key.split()
        if len(parts) < 2:
            continue

        tail = parts[-1]
        tail_entity = by_key.get(tail)
        if tail_entity is None:
            canonical_tail = source_to_canonical.get(tail)
            if canonical_tail is None:
                continue
            tail_entity = by_key.get(canonical_tail)
            if tail_entity is None:
                continue
            tail = canonical_tail
        if key in tail_entity.source_keys:
            continue

        anchor_paths = sorted({record.document_anchor.path for record in anchor_records})
        if not _is_safe_character_component(tail_entity, anchor_paths):
            continue

        if not all(
            part in title_prefixes_lower or has_generic_modifier_profile(part)
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
            merged_entities.append(build_corpus_entity(
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
