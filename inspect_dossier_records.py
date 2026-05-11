"""
Dossier record inspection entrypoint for the phase-1 non-manuscript experiment.

Usage:
    python inspect_dossier_records.py 'examples/story planning/estuary crew summaries.txt'

# Diagram omitted - this is a thin CLI entry point with sequential processing only.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_workspace = _os.path.dirname(_os.path.realpath(__file__))
if _workspace not in _sys.path:
    _sys.path.insert(0, _workspace)

from backend.nlp.experiments.dossier_review.cli import build_parser, run_dossier_review_experiment


def main() -> int:
    """Parse arguments and run the dossier review experiment.

    Returns:
        Shell exit status.
    """
    args = build_parser().parse_args()
    json_path, report_path, llm_report_path = run_dossier_review_experiment(
        args.input_path,
        args.output_dir,
        max_report_records=args.max_report_records,
        run_llm=args.run_llm,
        llm_model=args.llm_model,
        max_llm_records=args.max_llm_records,
        llm_timeout_seconds=args.llm_timeout_seconds,
    )
    print(f"Wrote dossier review JSON to {json_path}")
    print(f"Wrote dossier review report to {report_path}")
    print(f"Wrote dossier LLM report to {llm_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
