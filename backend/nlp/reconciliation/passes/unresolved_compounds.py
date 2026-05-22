"""
Unresolved compound deferral to stronger resolved anchors.

.. code-block:: mermaid

    flowchart TD
        A[CorpusEntity list + all exact-key records] --> B[Find longer unresolved compounds]
        B --> C[Resolve unique stronger non-character target from exact or absorbed aliases]
        C --> D[Defer longer unresolved compound into resolved anchor]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.reconciliation.passes.common import MergeChild, apply_merge_plan
from backend.nlp.types import CorpusEntity, DocumentEntityRecord, LexiconCategory


def defer_unresolved_longer_compounds_to_resolved_anchors(
    entities: list[CorpusEntity],
    all_records_by_key: dict[str, list[DocumentEntityRecord]],
) -> list[CorpusEntity]:
    """Defer weak longer unresolved compounds to stronger resolved anchors."""
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
        deferred_anchor_records[entity.canonical_key] = all_records_by_key.get(
            entity.canonical_key,
            entity.member_records,
        )
        deferred_children.add(entity.canonical_key)
        target_to_children[target_key].append(entity.canonical_key)

    merge_targets: dict[str, list[MergeChild]] = defaultdict(list)
    for target_key, child_keys in sorted(target_to_children.items()):
        for child_key in child_keys:
            merge_targets[target_key].append(MergeChild(
                child_key=child_key,
                anchor_records=deferred_anchor_records[child_key],
            ))

    return apply_merge_plan(
        entities,
        by_key,
        merge_targets,
        deferred_children,
        "longer unresolved compound deferred to stronger resolved non-character anchor",
    )
