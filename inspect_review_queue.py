"""
Review-queue artifact entrypoint for second-pass manuscript resolution prep.

Usage:
    python inspect_review_queue.py logs/manuscript-review/manuscript-corpus-report-baseline.json --llm-results logs/llm-tasks/manuscript-gptoss120b-refute-results.json

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

from backend.nlp.llm_tasks import build_manuscript_review_queue
from backend.nlp.llm_tasks.execution.io import (
    load_task_packets_from_artifact,
    load_task_results_from_artifact,
)
from backend.nlp.llm_tasks.review.review_queue_reports import render_review_queue_report
from backend.nlp.text_filtering import to_llm_safe_jsonable


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for review-queue artifact generation."""
    parser = argparse.ArgumentParser(
        description="Build second-pass review queue from first-pass manuscript LLM results.",
    )
    parser.add_argument(
        "artifact_path",
        help="Review-bundle handoff artifact JSON path.",
    )
    parser.add_argument(
        "--llm-results",
        required=True,
        help="First-pass shared LLM task result artifact JSON path.",
    )
    parser.add_argument(
        "--output",
        default="logs/review-queue/manuscript-review-queue.json",
        help="Output path for review-queue artifact JSON.",
    )
    parser.add_argument(
        "--max-snippets",
        type=int,
        default=5,
        help="Maximum evidence snippets per queue item. Default: 5",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=2000,
        help="Per-item context budget hint for pass-2 prompts. Default: 2000",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help=(
            "Optional human-readable report output path. "
            "Default: same path as --output with .txt suffix."
        ),
    )
    return parser


def main() -> int:
    """Build review-queue artifact from one handoff artifact and first-pass results."""
    args = _build_parser().parse_args()
    packets = load_task_packets_from_artifact(args.artifact_path)
    results = load_task_results_from_artifact(args.llm_results)
    queue_items = build_manuscript_review_queue(
        task_packets=packets,
        task_results=results,
        max_snippets=max(1, args.max_snippets),
        max_context_chars=max(100, args.max_context_chars),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_artifact_path": args.artifact_path,
        "source_llm_result_path": args.llm_results,
        "review_queue_version": "1",
        "review_queue_item_count": len(queue_items),
        "review_queue_items": to_llm_safe_jsonable(queue_items),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = (
        Path(args.report_output)
        if args.report_output
        else output_path.with_suffix(".txt")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_review_queue_report(queue_items),
        encoding="utf-8",
    )
    print(f"Wrote review queue artifact to {output_path}")
    print(f"Wrote review queue report to {report_path}")
    print(f"Review queue items: {len(queue_items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
