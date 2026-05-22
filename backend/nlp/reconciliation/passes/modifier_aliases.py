"""
Resolved non-character modifier alias reconciliation.

.. code-block:: mermaid

    flowchart TD
        A[CorpusEntity list + all exact-key records] --> B[Find resolved compounds with shorter leading modifier]
        B --> C[Require unique ownership, subset support, and frequency safety]
        C --> D[Merge shorter modifier alias into stronger compound]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.reconciliation.passes.common import MergeChild, apply_merge_plan
from backend.nlp.types import CorpusEntity, DocumentEntityRecord, LexiconCategory


def merge_non_character_modifier_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer shorter leading modifiers to stronger resolved compounds."""
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

        modifier_occ = sum(r.occurrence_count for r in modifier_entity.member_records)
        compound_occ = sum(r.occurrence_count for r in by_key[key].member_records)
        if modifier_occ > compound_occ:
            continue

        eligible_compounds[key] = (modifier, anchor_records)
        modifier_to_compounds[modifier].add(key)

    merge_targets: dict[str, list[MergeChild]] = defaultdict(list)
    absorbed_keys: set[str] = set()
    for key, (modifier, anchor_records) in sorted(eligible_compounds.items()):
        if modifier_to_compounds[modifier] != {key}:
            continue
        merge_targets[key].append(MergeChild(child_key=modifier, anchor_records=anchor_records))
        absorbed_keys.add(modifier)

    return apply_merge_plan(
        entities,
        by_key,
        merge_targets,
        absorbed_keys,
        "resolved non-character compound absorbed its shorter modifier alias",
    )
