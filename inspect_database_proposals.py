"""
Database proposal projection and validation entrypoint.

Usage:
    python inspect_database_proposals.py logs/structured-review/example-structured-review.json
    python inspect_database_proposals.py logs/manuscript-review/manuscript-corpus-report.json --llm-results logs/llm-tasks/llm-task-results.json

# Diagram omitted - this is a thin CLI entry point with sequential processing only.
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from pathlib import Path

_workspace = _os.path.dirname(_os.path.realpath(__file__))
if _workspace not in _sys.path:
    _sys.path.insert(0, _workspace)

from backend.nlp.indexing import (
    project_llm_task_results_to_database_proposals,
    project_task_packets_to_database_proposals,
    validate_database_proposals,
)
from backend.nlp.indexing.reports import render_database_proposal_report
from backend.nlp.llm_tasks.execution.io import (
    load_task_packets_from_artifact,
    load_task_results_from_artifact,
)
from backend.nlp.text_filtering import to_llm_safe_jsonable


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for database proposal projection."""
    parser = argparse.ArgumentParser(
        description="Project and validate database proposals from handoff and task-result artifacts.",
    )
    parser.add_argument(
        "artifact_paths",
        nargs="+",
        help="One or more review-bundle handoff artifact JSON files.",
    )
    parser.add_argument(
        "--llm-results",
        default=None,
        help="Optional shared LLM task result artifact JSON path.",
    )
    parser.add_argument(
        "--output",
        default="logs/indexing/database-proposals.json",
        help="Path to write the database proposal artifact JSON.",
    )
    parser.add_argument(
        "--report-output",
        default="logs/indexing/database-proposals.txt",
        help="Path to write the human-readable proposal report.",
    )
    return parser


def main() -> int:
    """Project deterministic and optional LLM proposals, then validate."""
    args = _build_parser().parse_args()
    task_packets = []
    for path in args.artifact_paths:
        task_packets.extend(load_task_packets_from_artifact(path))
    deterministic_proposals, deterministic_diagnostics = (
        project_task_packets_to_database_proposals(task_packets)
    )

    llm_proposals = []
    llm_diagnostics = []
    if args.llm_results:
        task_results = load_task_results_from_artifact(args.llm_results)
        llm_proposals, llm_diagnostics = project_llm_task_results_to_database_proposals(
            task_results,
            task_packets,
        )

    proposals, validation_diagnostics = validate_database_proposals(
        deterministic_proposals + llm_proposals
    )
    diagnostics = deterministic_diagnostics + llm_diagnostics + validation_diagnostics

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "source_artifact_paths": list(args.artifact_paths),
        "llm_result_path": args.llm_results or "",
        "database_proposals": to_llm_safe_jsonable(proposals),
        "indexing_diagnostics": to_llm_safe_jsonable(diagnostics),
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        render_database_proposal_report(proposals, diagnostics),
        encoding="utf-8",
    )

    print(f"Wrote database proposal artifact to {output_path}")
    print(f"Wrote database proposal report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
