"""
Corpus inspection tool - runs the manuscript NLP pipeline across multiple
documents and writes both a human-readable report and a machine-readable
manuscript handoff artifact.

Usage:
    python inspect_manuscript_corpus.py
    python inspect_manuscript_corpus.py --glob 'examples/*.md'
    python inspect_manuscript_corpus.py --glob 'examples/world context/*.txt'
    python inspect_manuscript_corpus.py --output logs/manuscript-review/my-report.txt

# Diagram omitted - this is a CLI entry point with sequential processing only.
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from pathlib import Path

# See backend/inspect.py for why this sys.path adjustment is needed when this
# file is run as a script from the repo root.
_workspace = _os.path.dirname(_os.path.realpath(__file__))
if _workspace not in _sys.path:
    _sys.path.insert(0, _workspace)

from backend.nlp.pipeline import run_document_pipeline
from backend.nlp.reconciliation.corpus_entities import reconcile_document_entities
from backend.nlp.llm_tasks import (
    build_rescue_task_packets,
    build_handoff_artifact,
    render_task_packet_report,
)
from backend.nlp.llm_tasks.io import (
    load_task_results_from_artifact,
    extract_rescued_keys_from_results,
)
from backend.nlp.semantic_review import (
    build_character_summaries,
    build_manuscript_review_bundle as build_manuscript_bundle,
    build_reference_clusters,
    build_conflict_records,
    build_review_tasks,
    manuscript_bundle_to_jsonable,
    render_manuscript_review_report,
)
from backend.nlp.harvesting.shared import TITLE_PREFIXES_LOWER
from backend.nlp.types import DocumentEntityRecord, ManuscriptReviewBundle, ReferenceCandidate


def _collect_document_outputs(
    paths: list[Path],
) -> tuple[list[DocumentEntityRecord], list[ReferenceCandidate], frozenset[str]]:
    """Run the existing document pipeline for each manuscript path.

    Args:
        paths: Source document paths to inspect.

    Returns:
        Document-local entity summaries, raw semantic reference candidates, and
        unioned induced title prefixes for the whole corpus run.
    """
    records: list[DocumentEntityRecord] = []
    references: list[ReferenceCandidate] = []
    induced_titles: set[str] = set()
    for path in paths:
        result = run_document_pipeline(str(path), path.read_text(encoding="utf-8"))
        records.extend(result.entity_records)
        references.extend(result.reference_candidates)
        induced_titles.update(result.bootstrap_result.induced_title_prefixes)
    return records, references, frozenset(induced_titles)


def _build_manuscript_review_bundle_with_rescue(
    paths: list[Path],
    *,
    rescue_results_path: str | None = None,
) -> tuple[ManuscriptReviewBundle, frozenset[str]]:
    """Build manuscript review bundle and optionally re-run reconciliation.

    Args:
        paths: Source document paths to inspect.
        rescue_results_path: Optional result artifact path from a completed
            suppression rescue LLM run.

    Returns:
        Review bundle plus the set of rescued normalized keys used for the
        optional second reconciliation pass.
    """
    records, references, induced_titles = _collect_document_outputs(paths)
    title_prefixes_lower = TITLE_PREFIXES_LOWER | frozenset(title.lower() for title in induced_titles)

    rescued_keys: frozenset[str] = frozenset()
    if rescue_results_path:
        result_path = Path(rescue_results_path)
        if result_path.exists():
            rescue_results = load_task_results_from_artifact(str(result_path))
            rescued_keys = extract_rescued_keys_from_results(rescue_results)

    corpus = reconcile_document_entities(
        records,
        title_prefixes_lower=title_prefixes_lower,
        rescued_keys=rescued_keys,
    )
    conflicts = build_conflict_records(corpus.canonical_entities)
    reference_clusters = build_reference_clusters(references, records)
    character_summaries = build_character_summaries(
        corpus.canonical_entities,
        reference_clusters,
        conflicts,
    )
    review_tasks = build_review_tasks(reference_clusters, conflicts, character_summaries)
    bundle = build_manuscript_bundle(
        paths,
        records,
        corpus.canonical_entities,
        references,
        reference_clusters,
        conflicts,
        character_summaries,
        review_tasks,
    )
    return bundle, rescued_keys


def _json_output_path(output_path: str, explicit_json_output: str | None) -> Path:
    """Return the JSON artifact path for the manuscript handoff bundle.

    Args:
        output_path: Human-readable report path.
        explicit_json_output: Optional explicit JSON output path.

    Returns:
        Path to the machine-readable manuscript artifact.
    """
    if explicit_json_output:
        return Path(explicit_json_output)

    report_path = Path(output_path)
    if report_path.suffix:
        return report_path.with_suffix(".json")
    return Path(str(report_path) + ".json")


def _write_manuscript_artifacts(
    bundle: ManuscriptReviewBundle,
    *,
    document_texts: dict[str, str],
    output_path: str,
    json_output_path: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write report and JSON manuscript handoff artifacts to disk.

    Args:
        bundle: Persisted manuscript handoff artifact.
        output_path: Destination path for the text report.
        json_output_path: Optional explicit JSON destination path.

    Returns:
        Paths to the written text report, JSON artifact, and task report.
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_manuscript_review_report(bundle), encoding="utf-8")

    task_packets, task_diagnostics = build_rescue_task_packets(
        bundle,
        document_texts,
    )

    artifact_path = _json_output_path(output_path, json_output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            build_handoff_artifact(
                source_kind="manuscript",
                review_bundle=manuscript_bundle_to_jsonable(bundle),
                task_packets=task_packets,
                task_diagnostics=task_diagnostics,
                extras={"document_paths": list(bundle.document_paths)},
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    task_report_path = artifact_path.with_name(
        artifact_path.stem + "-llm-task-packets.txt"
    )
    task_report_path.write_text(
        render_task_packet_report(
            task_packets,
            task_diagnostics,
            max_packets=12,
        ),
        encoding="utf-8",
    )
    return report_path, artifact_path, task_report_path


def main(
    glob_pattern: str,
    output_path: str,
    json_output_path: str | None = None,
    rescue_results_path: str | None = None,
) -> int:
    """Run manuscript corpus inspection and write both handoff artifacts.

    Args:
        glob_pattern: Glob for source document paths.
        output_path: Text report destination.
        json_output_path: Optional explicit JSON artifact destination.

    Returns:
        Process exit code.
    """
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        print(f"No files matched glob: {glob_pattern}", file=_sys.stderr)
        return 1

    document_texts = {
        str(path): path.read_text(encoding="utf-8")
        for path in paths
    }
    bundle, rescued_keys = _build_manuscript_review_bundle_with_rescue(
        paths,
        rescue_results_path=rescue_results_path,
    )
    report_path, artifact_path, task_report_path = _write_manuscript_artifacts(
        bundle,
        document_texts=document_texts,
        output_path=output_path,
        json_output_path=json_output_path,
    )

    print(f"Wrote corpus report for {len(paths)} documents to {report_path}")
    print(f"Wrote manuscript handoff artifact to {artifact_path}")
    print(f"Wrote manuscript LLM task report to {task_report_path}")
    if rescue_results_path:
        print(
            "Applied rescue-aware reconciliation from "
            f"{rescue_results_path} (rescued keys: {len(rescued_keys)})"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the manuscript corpus inspector.

    Returns:
        Configured CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run corpus reconciliation across document files and write a report."
    )
    parser.add_argument(
        "--glob",
        default="examples/*.md",
        help="Glob pattern for document files. Default: examples/*.md",
    )
    parser.add_argument(
        "--output",
        default="logs/manuscript-review/manuscript-corpus-report.txt",
        help="Path to write the text report. Default: logs/manuscript-review/manuscript-corpus-report.txt",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON artifact path. Default: sibling .json next to --output",
    )
    parser.add_argument(
        "--rescue-results",
        default=None,
        help=(
            "Optional LLM result artifact path. When provided, valid rescue "
            "verdicts are folded into reconciliation as rescued suppressed keys."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    raise SystemExit(main(args.glob, args.output, args.json_output, args.rescue_results))
