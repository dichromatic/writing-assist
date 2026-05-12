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
        E --> F[Write JSON artifact]
        E --> G[Write deterministic text report]
        E --> H[Write separate LLM text report]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.nlp.document_metadata import load_document_metadata_manifest, resolve_document_metadata
from backend.nlp.experiments.structured_review.claim_units import build_claim_units_from_review_bundles
from backend.nlp.experiments.structured_review.llm_pass import run_structured_llm_passes
from backend.nlp.experiments.structured_review.report import (
    render_structured_llm_report,
    render_structured_review_report,
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
    run_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    max_llm_records: int | None = None,
    llm_timeout_seconds: float = 60.0,
    metadata_manifest_path: str | None = None,
) -> tuple[Path, Path, Path]:
    """Run the deterministic structured review scaffold on one file.

    Args:
        input_path: Source .txt file path.
        output_dir: Directory that will receive JSON and text artifacts.
        max_report_records: Maximum full record blocks printed in the report.
        run_llm: Whether to run the first constrained LLM pass.
        llm_model: Model name to use for the live LLM pass.
        max_llm_records: Optional cap on how many bundles to send to the LLM.
        llm_timeout_seconds: Timeout for each live LLM request.
        metadata_manifest_path: Optional JSON sidecar manifest path.

    Returns:
        Paths to the JSON artifact, deterministic text report, and LLM text report.
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
    if run_llm:
        review_bundles = run_structured_llm_passes(
            review_bundles,
            model=llm_model,
            max_records=max_llm_records,
            timeout_seconds=llm_timeout_seconds,
        )
    claim_units = build_claim_units_from_review_bundles(review_bundles)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem.replace(" ", "-").lower()
    json_path = output_root / f"{stem}-structured-review.json"
    report_path = output_root / f"{stem}-structured-review.txt"
    llm_report_path = output_root / f"{stem}-structured-review-llm.txt"

    artifact = {
        "document_path": str(source_path),
        "document_metadata": to_llm_safe_jsonable(document_metadata),
        "diagnostics": to_llm_safe_jsonable(diagnostics),
        "structured_records": to_llm_safe_jsonable(structured_records),
        "review_bundles": to_llm_safe_jsonable(review_bundles),
        "claim_units": to_llm_safe_jsonable(claim_units),
    }
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        render_structured_review_report(diagnostics, review_bundles, max_records=max_report_records),
        encoding="utf-8",
    )
    llm_report_path.write_text(
        render_structured_llm_report(diagnostics, review_bundles, max_records=max_report_records),
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
        "--run-llm",
        action="store_true",
        help="Run the first constrained LLM pass for structured records.",
    )
    parser.add_argument(
        "--llm-model",
        default=(
            os.getenv("NVIDIA_MODEL")
            or os.getenv("NIM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        ),
        help="Model name for --run-llm. Default: env model override or gpt-4o-mini",
    )
    parser.add_argument(
        "--max-llm-records",
        type=int,
        default=None,
        help="Optional cap on structured records sent to the LLM.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request timeout for --run-llm. Default: 60",
    )
    parser.add_argument(
        "--metadata-manifest",
        default=None,
        help="Optional JSON sidecar manifest containing document status metadata.",
    )
    return parser
