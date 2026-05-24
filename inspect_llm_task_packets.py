"""
LLM task-packet runner - execute rescue task packets against an LLM endpoint.

Usage:
    python inspect_llm_task_packets.py logs/manuscript-review/manuscript-corpus-report.json
    python inspect_llm_task_packets.py logs/manuscript-review/manuscript-corpus-report.json --provider nim

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

from backend.nlp.llm_tasks.io import (
    load_task_packets_from_artifact,
    write_task_result_artifact,
)
from backend.nlp.llm_tasks.provider import (
    make_chat_responder,
    run_task_packets,
)
from backend.nlp.llm_tasks.reports import render_task_result_report
from backend.nlp.text_filtering import strip_emoji


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the task-packet runner."""
    parser = argparse.ArgumentParser(
        description="Run LLM task packets from one or more handoff artifacts.",
    )
    parser.add_argument(
        "artifact_paths",
        nargs="+",
        help="One or more handoff artifact JSON files.",
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
        default=(_os.getenv("LLM_PROVIDER") or "openai"),
        help="Provider identifier: dry_run, nim, or openai. Default: openai.",
    )
    parser.add_argument(
        "--model",
        default=(
            _os.getenv("LOCAL_MODEL")
            or _os.getenv("OPENAI_MODEL")
            or
            _os.getenv("NIM_MODEL")
            or _os.getenv("NVIDIA_MODEL")
            or "Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound"
        ),
        help="Model identifier for traceability.",
    )
    parser.add_argument(
        "--base-url",
        default=(
            _os.getenv("LOCAL_BASE_URL")
            or _os.getenv("OPENAI_BASE_URL")
            or
            _os.getenv("NIM_BASE_URL")
            or "http://host.docker.internal:8001/v1"
        ),
        help="Chat completions API base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature. Default: 0.3",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p. Default: 0.9",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum response tokens. Default: 4096",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="Optional path for a human-readable result report.",
    )
    return parser


def _render_summary(results) -> str:
    """Render a short result summary for console output."""
    completed = sum(1 for r in results if r.status.value == "completed")
    failed = sum(1 for r in results if r.status.value == "failed")
    skipped = sum(1 for r in results if r.status.value == "skipped")
    lines = [
        "LLM TASK RUN SUMMARY",
        f"  total: {len(results)}",
        f"  completed: {completed}",
        f"  failed: {failed}",
        f"  skipped: {skipped}",
    ]
    return strip_emoji("\n".join(lines))


def main() -> int:
    """Run the LLM task-packet pipeline."""
    args = _build_parser().parse_args()
    all_packets = []
    for path in args.artifact_paths:
        all_packets.extend(load_task_packets_from_artifact(path))
    if args.max_tasks is not None:
        all_packets = all_packets[: max(0, args.max_tasks)]

    responder = None
    provider = args.provider.strip().casefold()
    if provider in {"nvidia_nim", "nim", "openai_compatible", "openai_compat", "openai"}:
        api_key = (
            _os.getenv("NVIDIA_API_KEY")
            or _os.getenv("NIM_API_KEY")
            or _os.getenv("OPENAI_API_KEY")
            or ""
        )
        if not api_key:
            print(
                "Set NVIDIA_API_KEY, NIM_API_KEY, or OPENAI_API_KEY.",
                file=_sys.stderr,
            )
            return 2
        responder = make_chat_responder(
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output = (
        Path(args.report_output)
        if args.report_output
        else output_path.with_name(f"{output_path.stem}-report.txt")
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)

    rolling_results: list = []

    def _persist_progress(result) -> None:
        """Persist artifact after each task result for long-running visibility."""
        rolling_results.append(result)
        write_task_result_artifact(
            output_path=str(output_path),
            source_artifact_paths=list(args.artifact_paths),
            packets=all_packets,
            results=rolling_results,
        )
        report_output.write_text(
            render_task_result_report(all_packets, rolling_results),
            encoding="utf-8",
        )

    results = run_task_packets(
        all_packets,
        model=args.model,
        provider=args.provider,
        responder=responder,
        on_result=_persist_progress,
    )

    write_task_result_artifact(
        output_path=str(output_path),
        source_artifact_paths=list(args.artifact_paths),
        packets=all_packets,
        results=results,
    )
    report_output.write_text(
        render_task_result_report(all_packets, results),
        encoding="utf-8",
    )
    print(_render_summary(results))
    print(f"Wrote task results to {output_path}")
    print(f"Wrote result report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
