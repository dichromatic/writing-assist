"""
Structured review experiment CLI - runs deterministic structured-note scaffolding.

.. code-block:: mermaid

    flowchart TD
        A[Input .txt path] --> B[Parse and preprocess full document]
        B --> D[Segment StructuredRecord list]
        D --> C[Extract structured entity inventory]
        M[Optional metadata manifest] --> N[Resolve document metadata]
        A --> N
        B --> N
        C & D & N --> E[Build RecordReviewBundle list]
        E --> F[Write JSON artifact]
        E --> G[Write deterministic text report]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.nlp.document_metadata import load_document_metadata_manifest, resolve_document_metadata
from backend.nlp.experiments.structured_review.report import render_structured_review_report
from backend.nlp.experiments.structured_review.review_bundle import build_structured_review_bundles
from backend.nlp.parsing.document_parser import parse
from backend.nlp.structured_records import extract_structural_entities, segment_structured_records
from backend.nlp.text_filtering import to_llm_safe_jsonable


def run_structured_review_experiment(
    input_path: str,
    output_dir: str,
    *,
    max_report_records: int,
    metadata_manifest_path: str | None = None,
) -> tuple[Path, Path]:
    """Run the deterministic structured review scaffold on one file.

    Args:
        input_path: Source .txt file path.
        output_dir: Directory that will receive JSON and text artifacts.
        max_report_records: Maximum full record blocks printed in the report.
        metadata_manifest_path: Optional JSON sidecar manifest path.

    Returns:
        Paths to the JSON artifact and deterministic text report.
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
    structured_records = segment_structured_records(doc)
    entity_inventory = extract_structural_entities(structured_records)
    review_bundles, diagnostics = build_structured_review_bundles(
        structured_records,
        entity_inventory=entity_inventory,
        document_metadata=document_metadata,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem.replace(" ", "-").lower()
    json_path = output_root / f"{stem}-structured-review.json"
    report_path = output_root / f"{stem}-structured-review.txt"

    artifact = {
        "artifact_version": "2",
        "source_kind": "structured_record",
        "document_path": str(source_path),
        "document_metadata": to_llm_safe_jsonable(document_metadata),
        "diagnostics": to_llm_safe_jsonable(diagnostics),
        "review_bundles": to_llm_safe_jsonable(review_bundles),
        "structured_records": to_llm_safe_jsonable(structured_records),
        "structured_entity_inventory": {
            "mentions": to_llm_safe_jsonable(entity_inventory.mentions),
            "names": sorted(entity_inventory.names),
            "mentions_by_record": to_llm_safe_jsonable(entity_inventory.mentions_by_record),
            "records_by_name": to_llm_safe_jsonable(entity_inventory.records_by_name),
        },
    }
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        render_structured_review_report(diagnostics, review_bundles, max_records=max_report_records),
        encoding="utf-8",
    )
    return json_path, report_path


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
