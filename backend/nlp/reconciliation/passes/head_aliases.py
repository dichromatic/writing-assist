"""
Resolved non-character head alias reconciliation.

.. code-block:: mermaid

    flowchart TD
        A[CorpusEntity list + all exact-key records] --> B[Find resolved compounds with shorter head alias]
        B --> C[Require unique ownership and category compatibility]
        C --> D[Merge shorter head alias into stronger compound]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.reconciliation.passes.common import MergeChild, apply_merge_plan
from backend.nlp.types import CorpusEntity, DocumentEntityRecord, LexiconCategory


def _is_safe_non_character_head_component(
    component: CorpusEntity,
    compound_category: LexiconCategory,
    supporting_document_paths: list[str],
) -> bool:
    """Return True when a trailing head key can defer to a stronger compound."""
    if component.review_required and component.dominant_category != compound_category:
        return False
    if component.dominant_category not in {compound_category, LexiconCategory.UNRESOLVED}:
        return False
    return set(component.supporting_document_paths).issubset(set(supporting_document_paths))


def merge_non_character_head_aliases(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer shorter head keys to stronger resolved non-character compounds."""
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

        anchor_paths = sorted({record.identity.document_anchor.path for record in anchor_records})
        if not _is_safe_non_character_head_component(
            head_entity,
            compound_entity.dominant_category,
            anchor_paths,
        ):
            continue

        eligible_compounds[key] = (head, anchor_records)
        head_to_compounds[head].add(key)

    merge_targets: dict[str, list[MergeChild]] = defaultdict(list)
    absorbed_keys: set[str] = set()
    for key, (head, anchor_records) in sorted(eligible_compounds.items()):
        if head_to_compounds[head] != {key}:
            continue
        merge_targets[key].append(MergeChild(child_key=head, anchor_records=anchor_records))
        absorbed_keys.add(head)

    return apply_merge_plan(
        entities,
        by_key,
        merge_targets,
        absorbed_keys,
        "resolved non-character compound absorbed its shorter head alias",
    )
