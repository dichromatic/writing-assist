"""
Corpus inspection tool - runs the manuscript NLP pipeline across multiple
documents, reconciles exact-key entities across the corpus, and writes a
human-readable report to disk.

Usage:
    python inspect_manuscript_corpus.py
    python inspect_manuscript_corpus.py --glob 'examples/*.md'
    python inspect_manuscript_corpus.py --output logs/my-report.txt

# Diagram omitted - this is a CLI entry point with sequential processing only.
"""

from __future__ import annotations

import argparse
import os as _os
import sys as _sys
from pathlib import Path

# See backend/inspect.py for why this sys.path adjustment is needed when this
# file is run as a script from the repo root.
_workspace = _os.path.dirname(_os.path.realpath(__file__))
if _workspace not in _sys.path:
    _sys.path.insert(0, _workspace)

from backend.nlp.lexicon.bootstrap import bootstrap
from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.promotion.attribution import attribute_dialogue
from backend.nlp.promotion.promotion import promote
from backend.nlp.reconciliation.corpus_entities import reconcile_document_entities
from backend.nlp.reconciliation.document_entities import summarize_document_entities
from backend.nlp.semantic_review import (
    build_character_summaries,
    build_reference_clusters,
    build_conflict_records,
    build_review_tasks,
    extract_reference_candidates,
)
from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictRecord,
    CorpusEntity,
    DocumentEntityBucket,
    DocumentEntityRecord,
    LexiconCategory,
    ReferenceCandidate,
    ReferenceCluster,
    ReferenceCandidateType,
    ReviewTask,
)


def _hr(title: str = "") -> str:
    width = 72
    if title:
        pad = width - len(title) - 2
        return f"\n-- {title} " + "-" * pad
    return "-" * width


def _collect_document_outputs(
    paths: list[Path],
) -> tuple[list[DocumentEntityRecord], list[ReferenceCandidate]]:
    """Run the existing document pipeline for each manuscript path."""
    records: list[DocumentEntityRecord] = []
    references: list[ReferenceCandidate] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        doc = parse(str(path), raw)
        pre = preprocess(doc)
        result = bootstrap(doc)
        attribution_records = attribute_dialogue(pre, result.clusters)
        bundle = promote(pre, result.clusters, result.lexicon, attribution_records)
        document_records = summarize_document_entities(pre, result.clusters, attribution_records, bundle)
        records.extend(document_records)
        references.extend(extract_reference_candidates(pre, document_records, attribution_records))
    return records, references


def _format_entity_line(entity: CorpusEntity) -> str:
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
    return (
        f"  {entity.canonical_key:20s}  {entity.dominant_category.value:10s}"
        f"  docs={docs:<2d}  conf={entity.aggregate_confidence:.3f}  {status}{conflicts}{aliases}"
    )


def _category_sort_key(category: LexiconCategory) -> tuple[int, str]:
    """Return a stable presentation order for corpus entity categories."""
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


def _format_report(
    paths: list[Path],
    records: list[DocumentEntityRecord],
    corpus,
    references: list[ReferenceCandidate],
    reference_clusters: list[ReferenceCluster],
    conflicts: list[ConflictRecord],
    character_summaries: list[CharacterSemanticSummary],
    review_tasks: list[ReviewTask],
) -> str:
    """Render a stable human-readable corpus report."""
    promoted_count = sum(1 for record in records if record.bucket == DocumentEntityBucket.PROMOTED)
    review_count = sum(1 for record in records if record.bucket == DocumentEntityBucket.REVIEW_ONLY)
    suppressed_count = sum(1 for record in records if record.bucket == DocumentEntityBucket.SUPPRESSED)

    lines: list[str] = []
    lines.append(_hr("CORPUS"))
    lines.append(f"  Manuscripts       : {len(paths)}")
    lines.append(f"  Entity records    : {len(records)}")
    lines.append(f"  Canonical entities: {len(corpus.canonical_entities)}")
    lines.append(f"  Promoted records  : {promoted_count}")
    lines.append(f"  Review-only       : {review_count}")
    lines.append(f"  Suppressed        : {suppressed_count}")

    lines.append(_hr("FILES"))
    for path in paths:
        lines.append(f"  {path}")

    sorted_entities = sorted(
        corpus.canonical_entities,
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
    review_entities = [entity for entity in corpus.canonical_entities if entity.review_required]
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
    for reference in references:
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
    lines.append(f"  grouped_reference_clusters  count={len(reference_clusters)}")
    for reference in sorted(
        reference_clusters,
        key=lambda item: (
            item.reference_type.value,
            item.document_anchor.path,
            -item.occurrence_count,
            item.normalized,
        ),
    )[:40]:
        links = ",".join(reference.candidate_entity_scores) if reference.candidate_entity_scores else "-"
        speakers = ",".join(reference.speaker_entity_scores) if reference.speaker_entity_scores else "-"
        lines.append(
            f"  {reference.reference_type.value:20s}  {reference.normalized:12s}"
            f"  path={reference.document_anchor.path}  occ={reference.occurrence_count:<2d}"
            f"  addr={reference.address_like_count:<2d}  speakers={speakers}  links={links}"
        )

    lines.append(_hr("SEMANTIC CONFLICTS"))
    if conflicts:
        for conflict in conflicts:
            categories = ",".join(category.value for category in conflict.conflicting_categories)
            lines.append(
                f"  {conflict.canonical_key:20s}  {conflict.source.value:26s}  "
                f"categories={categories}"
            )
            lines.append(f"    reason: {conflict.reason}")
    else:
        lines.append("  None.")

    lines.append(_hr("CHARACTER SEMANTIC SUMMARIES"))
    for summary in character_summaries[:40]:
        aliases = ",".join(summary.alias_keys) if summary.alias_keys else "-"
        attached = ",".join(
            f"{title}:{count}" for title, count in summary.attached_title_counts.items()
        ) or "-"
        ambiguous = ",".join(
            f"{title}:{count}" for title, count in summary.ambiguous_title_counts.items()
        ) or "-"
        conflicts_text = ",".join(source.value for source in summary.conflict_sources) or "-"
        attached_relations = ",".join(
            f"{title}:{count}" for title, count in summary.attached_relation_counts.items()
        ) or "-"
        ambiguous_relations = ",".join(
            f"{title}:{count}" for title, count in summary.ambiguous_relation_counts.items()
        ) or "-"
        lines.append(
            f"  {summary.canonical_key:20s}  docs={len(summary.supporting_document_paths):<2d}"
            f"  attr={summary.aggregate_attribution_count:<2d}  aliases={aliases}"
        )
        lines.append(f"    attached_titles: {attached}")
        lines.append(f"    ambiguous_titles: {ambiguous}")
        lines.append(f"    attached_relations: {attached_relations}")
        lines.append(f"    ambiguous_relations: {ambiguous_relations}")
        lines.append(f"    conflict_sources: {conflicts_text}")

    lines.append(_hr("SEMANTIC REVIEW TASKS"))
    for task in review_tasks[:40]:
        lines.append(
            f"  {task.kind.value:20s}  {task.subject_key:20s}  "
            f"paths={', '.join(task.supporting_anchor_paths)}"
        )
        lines.append(f"    prompt: {task.prompt}")

    lines.append(_hr("TOP DOCUMENT SUPPORT"))
    for entity in sorted(
        corpus.canonical_entities,
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
    return "\n".join(lines) + "\n"


def main(glob_pattern: str, output_path: str) -> int:
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        print(f"No files matched glob: {glob_pattern}", file=_sys.stderr)
        return 1

    records, references = _collect_document_outputs(paths)
    corpus = reconcile_document_entities(records)
    conflicts = build_conflict_records(corpus.canonical_entities)
    reference_clusters = build_reference_clusters(references)
    character_summaries = build_character_summaries(
        corpus.canonical_entities,
        reference_clusters,
        conflicts,
    )
    review_tasks = build_review_tasks(reference_clusters, conflicts, character_summaries)
    report = _format_report(
        paths,
        records,
        corpus,
        references,
        reference_clusters,
        conflicts,
        character_summaries,
        review_tasks,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(f"Wrote corpus report for {len(paths)} manuscripts to {output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run corpus reconciliation across manuscript files and write a report."
    )
    parser.add_argument(
        "--glob",
        default="examples/*.md",
        help="Glob pattern for manuscript files. Default: examples/*.md",
    )
    parser.add_argument(
        "--output",
        default="logs/manuscript-corpus-report.txt",
        help="Path to write the text report. Default: logs/manuscript-corpus-report.txt",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    raise SystemExit(main(args.glob, args.output))
