"""
Structured review experiment CLI - runs deterministic structured-note scaffolding.

.. code-block:: mermaid

    flowchart TD
        A[Input .txt path] --> B[Parse and preprocess full document]
        B --> C[Run existing document extraction as weak hints]
        B --> D[Segment StructuredRecord list]
        M[Optional metadata manifest] --> N[Resolve document metadata]
        A --> N
        B --> N
        C & D & N --> E[Build RecordReviewBundle list]
        E --> F[Build shared LLMTaskPacket list]
        F --> G[Write JSON artifact]
        F --> H[Write deterministic text report]
        F --> I[Write task-packet text report]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.nlp.document_metadata import load_document_metadata_manifest, resolve_document_metadata
from backend.nlp.experiments.structured_review.report import render_structured_review_report
from backend.nlp.llm_tasks import (
    build_llm_task_packets,
    build_review_bundle_handoff_artifact,
    render_llm_task_packet_report,
)
from backend.nlp.experiments.structured_review.review_bundle import build_structured_review_bundles
from backend.nlp.pipeline import run_document_pipeline
from backend.nlp.parsing.document_parser import parse
from backend.nlp.structured_records import segment_structured_records
from backend.nlp.text_filtering import to_llm_safe_jsonable


def run_structured_review_experiment(
    input_path: str,
    output_dir: str,
    *,
    max_report_records: int,
    metadata_manifest_path: str | None = None,
) -> tuple[Path, Path, Path]:
    """Run the deterministic structured review scaffold on one file.

    Args:
        input_path: Source .txt file path.
        output_dir: Directory that will receive JSON and text artifacts.
        max_report_records: Maximum full record blocks printed in the report.
        metadata_manifest_path: Optional JSON sidecar manifest path.

    Returns:
        Paths to the JSON artifact, deterministic text report, and task report.
    """
    source_path = Path(input_path)
    raw_text = source_path.read_text(encoding="utf-8")
    metadata_manifest = load_document_metadata_manifest(metadata_manifest_path)
    document_metadata = resolve_document_metadata(
        str(source_path),
        raw_text,
        metadata_manifest,
    )
    doc = parse(str(source_path), raw_text)
    pipeline = run_document_pipeline(str(source_path), raw_text)
    structured_records = segment_structured_records(doc)
    review_bundles, diagnostics = build_structured_review_bundles(
        structured_records,
        pipeline.entity_records,
        pipeline.reference_candidates,
        document_metadata,
    )
    llm_task_packets, llm_task_diagnostics = build_llm_task_packets(
        record_review_bundles=review_bundles
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem.replace(" ", "-").lower()
    json_path = output_root / f"{stem}-structured-review.json"
    report_path = output_root / f"{stem}-structured-review.txt"
    llm_report_path = output_root / f"{stem}-structured-review-llm-task-packets.txt"

    artifact = build_review_bundle_handoff_artifact(
        source_kind="structured_record",
        review_bundle_kind="record_review_bundle_list",
        review_bundle=review_bundles,
        llm_task_packets=llm_task_packets,
        llm_task_diagnostics=llm_task_diagnostics,
        extras={
            "document_path": str(source_path),
            "document_metadata": to_llm_safe_jsonable(document_metadata),
            "diagnostics": to_llm_safe_jsonable(diagnostics),
            "structured_records": to_llm_safe_jsonable(structured_records),
        },
    )
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        render_structured_review_report(diagnostics, review_bundles, max_records=max_report_records),
        encoding="utf-8",
    )
    llm_report_path.write_text(
        render_llm_task_packet_report(
            llm_task_packets,
            llm_task_diagnostics,
            max_packets=max_report_records,
        ),
        encoding="utf-8",
    )
    return json_path, report_path, llm_report_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the structured review experiment.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run the deterministic structured-review scaffold on one .txt file."
    )
    parser.add_argument("input_path", help="Path to one .txt note file.")
    parser.add_argument(
        "--output-dir",
        default="logs/structured-review",
        help="Directory for JSON and text artifacts. Default: logs/structured-review",
    )
    parser.add_argument(
        "--max-report-records",
        type=int,
        default=5,
        help="Maximum full record blocks in the text report. Default: 5",
    )
    parser.add_argument(
        "--metadata-manifest",
        default=None,
        help="Optional JSON sidecar manifest containing document status metadata.",
    )
    return parser
