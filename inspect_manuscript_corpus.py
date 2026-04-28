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
from backend.nlp.types import (
    CorpusEntity,
    DocumentEntityBucket,
    DocumentEntityRecord,
    LexiconCategory,
)


def _hr(title: str = "") -> str:
    width = 72
    if title:
        pad = width - len(title) - 2
        return f"\n-- {title} " + "-" * pad
    return "-" * width


def _collect_document_records(paths: list[Path]) -> list[DocumentEntityRecord]:
    """Run the existing document pipeline for each manuscript path."""
    records: list[DocumentEntityRecord] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        doc = parse(str(path), raw)
        pre = preprocess(doc)
        result = bootstrap(doc)
        attribution_records = attribute_dialogue(pre, result.clusters)
        bundle = promote(pre, result.clusters, result.lexicon, attribution_records)
        records.extend(
            summarize_document_entities(pre, result.clusters, attribution_records, bundle)
        )
    return records


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

    records = _collect_document_records(paths)
    corpus = reconcile_document_entities(records)
    report = _format_report(paths, records, corpus)

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
