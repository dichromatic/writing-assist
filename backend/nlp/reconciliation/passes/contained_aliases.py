"""
Resolved non-character contained alias reconciliation.

.. code-block:: mermaid

    flowchart TD
        A[CorpusEntity list + all exact-key records] --> B[Find resolved compounds with contained suffix alias]
        B --> C[Require multi-token alias, unique ownership, and frequency safety]
        C --> D[Merge shorter contained alias into stronger compound]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.reconciliation.passes.common import MergeChild, apply_merge_plan
from backend.nlp.types import CorpusEntity, DocumentEntityRecord, LexiconCategory


def merge_non_character_contained_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer shorter contained compounds to stronger resolved compounds."""
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

            alias_occ = sum(
                record.occurrence_count
                for record in all_records_by_key.get(alias_key, [])
            )
            compound_occ = sum(
                record.occurrence_count
                for record in all_records_by_key.get(key, [])
            )
            if alias_occ > compound_occ:
                continue

            eligible_compounds[key].append((alias_key, anchor_records))
            alias_to_compounds[alias_key].add(key)

    merge_targets: dict[str, list[MergeChild]] = defaultdict(list)
    absorbed_keys: set[str] = set()
    for key, alias_entries in sorted(eligible_compounds.items()):
        for alias_key, anchor_records in alias_entries:
            if alias_to_compounds[alias_key] != {key}:
                continue
            merge_targets[key].append(MergeChild(child_key=alias_key, anchor_records=anchor_records))
            absorbed_keys.add(alias_key)

    return apply_merge_plan(
        entities,
        by_key,
        merge_targets,
        absorbed_keys,
        "resolved non-character compound absorbed its shorter contained alias",
    )
