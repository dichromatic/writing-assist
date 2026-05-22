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

from backend.nlp.harvesting.shared import TITLE_PREFIXES_LOWER
from backend.nlp.reconciliation.passes import (
    defer_unresolved_longer_compounds_to_resolved_anchors,
    merge_character_compound_aliases,
    merge_generic_leading_character_aliases,
    merge_non_character_contained_aliases,
    merge_non_character_head_aliases,
    merge_non_character_modifier_aliases,
)
from backend.nlp.reconciliation.passes.common import build_corpus_entity
from backend.nlp.types import (
    CorpusReconciliationResult,
    DocumentEntityBucket,
    DocumentEntityRecord,
)

def reconcile_document_entities(
    records: list[DocumentEntityRecord],
    *,
    include_suppressed: bool = False,
    title_prefixes_lower: frozenset[str] = TITLE_PREFIXES_LOWER,
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
        exact_entities.append(build_corpus_entity(
            canonical_key=key,
            source_keys=[key],
            members=members,
            reasons=["exact normalized key matched across documents"],
        ))

    canonical_entities = merge_character_compound_aliases(
        exact_entities,
        all_grouped,
        title_prefixes_lower=title_prefixes_lower,
    )
    canonical_entities = merge_generic_leading_character_aliases(
        canonical_entities,
        all_grouped,
        title_prefixes_lower=title_prefixes_lower,
    )
    canonical_entities = merge_non_character_head_aliases(canonical_entities, all_grouped)
    canonical_entities = merge_non_character_modifier_aliases(canonical_entities, all_grouped)
    canonical_entities = merge_non_character_contained_aliases(canonical_entities, all_grouped)
    canonical_entities = defer_unresolved_longer_compounds_to_resolved_anchors(
        canonical_entities,
        all_grouped,
    )
    return CorpusReconciliationResult(
        canonical_entities=sorted(canonical_entities, key=lambda entity: entity.canonical_key)
    )
