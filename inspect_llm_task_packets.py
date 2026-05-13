"""
LLM task-packet runner entrypoint.

Usage:
    python inspect_llm_task_packets.py logs/structured-review/example-structured-review.json
    python inspect_llm_task_packets.py logs/manuscript-review/manuscript-corpus-report.json

# Diagram omitted - this is a thin CLI entry point with sequential processing only.
"""

from __future__ import annotations

import argparse
import os as _os
import sys as _sys
from pathlib import Path

_workspace = _os.path.dirname(_os.path.realpath(__file__))
if _workspace not in _sys.path:
    _sys.path.insert(0, _workspace)

from backend.nlp.llm_tasks.execution.io import (
    load_task_packets_from_artifact,
    write_task_result_artifact,
)
from backend.nlp.llm_tasks.execution.provider import (
    make_nvidia_nim_chat_responder,
    run_llm_task_packets,
)
from backend.nlp.llm_tasks.execution.reports import (
    render_llm_task_result_comparison_report,
)
from backend.nlp.llm_tasks.review.review_resolution import (
    build_review_resolution_task_packets,
)
from backend.nlp.types import LLMTaskPassStage
from backend.nlp.text_filtering import strip_emoji


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the shared task-packet runner."""
    parser = argparse.ArgumentParser(
        description="Run shared LLM task packets from one or more handoff artifacts.",
    )
    parser.add_argument(
        "artifact_paths",
        nargs="+",
        help="One or more review-bundle handoff artifact JSON files.",
    )
    parser.add_argument(
        "--output",
        default="logs/llm-tasks/llm-task-results.json",
        help="Path to write the task-result artifact.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on how many task packets to execute.",
    )
    parser.add_argument(
        "--provider",
        default="dry_run",
        help="Provider identifier for traceability in the result artifact.",
    )
    parser.add_argument(
        "--model",
        default=(
            _os.getenv("NIM_MODEL")
            or _os.getenv("NVIDIA_MODEL")
            or "dry_run_model"
        ),
        help="Model identifier for traceability in the result artifact.",
    )
    parser.add_argument(
        "--nim-base-url",
        default=(
            _os.getenv("NIM_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        ),
        help="NVIDIA NIM base URL. Default: env NIM_BASE_URL or NVIDIA endpoint.",
    )
    parser.add_argument(
        "--nim-temperature",
        type=float,
        default=0.6,
        help="NVIDIA NIM chat completion temperature. Default: 0.6",
    )
    parser.add_argument(
        "--nim-top-p",
        type=float,
        default=0.9,
        help="NVIDIA NIM chat completion top_p. Default: 0.9",
    )
    parser.add_argument(
        "--nim-max-tokens",
        type=int,
        default=4096,
        help="NVIDIA NIM chat completion max_tokens. Default: 4096",
    )
    parser.add_argument(
        "--nim-timeout-seconds",
        type=float,
        default=60.0,
        help="NVIDIA NIM request timeout in seconds. Default: 60",
    )
    parser.add_argument(
        "--prompt-variant",
        default="baseline",
        choices=["baseline", "downgraded", "refute_first"],
        help=(
            "Prompt strategy variant for provider-side system prompts. "
            "Default: baseline."
        ),
    )
    parser.add_argument(
        "--report-output",
        default="",
        help=(
            "Optional path for a human-readable deterministic-vs-LLM comparison "
            "report. Default: derived from --output with .txt suffix."
        ),
    )
    parser.add_argument(
        "--review-resolution",
        action="store_true",
        help="Run sequential second-pass review-resolution tasks after first pass.",
    )
    parser.add_argument(
        "--max-review-resolution-tasks",
        type=int,
        default=None,
        help="Optional cap on second-pass review-resolution tasks.",
    )
    return parser


def _render_summary(results) -> str:
    """Render a short result summary for console output."""
    completed = sum(1 for item in results if item.status.value == "completed")
    failed = sum(1 for item in results if item.status.value == "failed")
    skipped = sum(1 for item in results if item.status.value == "skipped")
    lines = [
        "LLM TASK RUN SUMMARY",
        f"  total: {len(results)}",
        f"  completed: {completed}",
        f"  failed: {failed}",
        f"  skipped: {skipped}",
    ]
    return strip_emoji("\n".join(lines))


def main() -> int:
    """Run the shared LLM task-packet pipeline in dry-run mode by default."""
    args = _build_parser().parse_args()
    all_packets = []
    for path in args.artifact_paths:
        all_packets.extend(load_task_packets_from_artifact(path))
    if args.max_tasks is not None:
        all_packets = all_packets[: max(0, args.max_tasks)]

    responder = None
    provider = args.provider.strip().casefold()
    if provider in {"nvidia_nim", "nim"}:
        api_key = _os.getenv("NVIDIA_API_KEY") or _os.getenv("NIM_API_KEY") or ""
        if not api_key:
            print("NVIDIA_API_KEY or NIM_API_KEY is not set.", file=_sys.stderr)
            return 2
        responder = make_nvidia_nim_chat_responder(
            api_key=api_key,
            base_url=args.nim_base_url,
            temperature=args.nim_temperature,
            top_p=args.nim_top_p,
            max_tokens=args.nim_max_tokens,
            timeout_seconds=args.nim_timeout_seconds,
            prompt_variant=args.prompt_variant,
        )

    first_pass_results = run_llm_task_packets(
        all_packets,
        model=args.model,
        provider=args.provider,
        responder=responder,
        pass_stage=LLMTaskPassStage.FIRST_PASS,
    )
    results = list(first_pass_results)
    if args.review_resolution:
        rr_packets, _rr_diagnostics = build_review_resolution_task_packets(
            first_pass_packets=all_packets,
            first_pass_results=first_pass_results,
            max_tasks=args.max_review_resolution_tasks,
        )
        if rr_packets:
            rr_results = run_llm_task_packets(
                rr_packets,
                model=args.model,
                provider=args.provider,
                responder=responder,
                pass_stage=LLMTaskPassStage.REVIEW_RESOLUTION,
            )
            results.extend(rr_results)
            all_packets = all_packets + rr_packets
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_task_result_artifact(
        output_path=str(output_path),
        source_artifact_paths=list(args.artifact_paths),
        packets=all_packets,
        results=results,
    )
    report_output = (
        Path(args.report_output)
        if args.report_output
        else output_path.with_name(f"{output_path.stem}-comparison.txt")
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        render_llm_task_result_comparison_report(all_packets, results),
        encoding="utf-8",
    )
    print(_render_summary(results))
    print(f"Wrote shared LLM task results to {output_path}")
    print(f"Wrote LLM comparison report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
