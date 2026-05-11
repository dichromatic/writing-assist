"""
Dossier review experiment CLI - runs phase-1 deterministic dossier scaffolding.

.. code-block:: mermaid

    flowchart TD
        A[Input .txt path] --> B[Parse and preprocess full document]
        B --> C[Run existing document extraction as weak hints]
        B --> D[Segment StructuredRecord list]
        C & D --> E[Build RecordReviewBundle list]
        E --> F[Write JSON artifact]
        E --> G[Write deterministic text report]
        E --> H[Write separate LLM text report]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.nlp.experiments.dossier_review.llm_pass import run_dossier_llm_passes
from backend.nlp.experiments.dossier_review.report import (
    render_dossier_llm_report,
    render_dossier_review_report,
)
from backend.nlp.experiments.dossier_review.review_bundle import build_dossier_review_bundles
from backend.nlp.lexicon.bootstrap import bootstrap
from backend.nlp.parsing.document_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.promotion.attribution import attribute_dialogue
from backend.nlp.promotion.promotion import promote
from backend.nlp.reconciliation.document_entities import summarize_document_entities
from backend.nlp.semantic_review import extract_reference_candidates
from backend.nlp.structured_records import segment_structured_records
from backend.nlp.text_filtering import to_llm_safe_jsonable


def run_dossier_review_experiment(
    input_path: str,
    output_dir: str,
    *,
    max_report_records: int,
    run_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    max_llm_records: int | None = None,
    llm_timeout_seconds: float = 60.0,
) -> tuple[Path, Path, Path]:
    """Run the phase-1 deterministic dossier review scaffold on one file.

    Args:
        input_path: Source .txt file path.
        output_dir: Directory that will receive JSON and text artifacts.
        max_report_records: Maximum full record blocks printed in the report.
        run_llm: Whether to run the first constrained LLM pass.
        llm_model: Model name to use for the live LLM pass.
        max_llm_records: Optional cap on how many bundles to send to the LLM.
        llm_timeout_seconds: Timeout for each live LLM request.

    Returns:
        Paths to the JSON artifact, deterministic text report, and LLM text report.
    """
    source_path = Path(input_path)
    raw_text = source_path.read_text(encoding="utf-8")
    doc = parse(str(source_path), raw_text)
    pre = preprocess(doc)
    result = bootstrap(doc)
    attribution_records = attribute_dialogue(pre, result.clusters)
    bundle = promote(pre, result.clusters, result.lexicon, attribution_records)
    entity_records = summarize_document_entities(pre, result.clusters, attribution_records, bundle)
    reference_candidates = extract_reference_candidates(pre, entity_records, attribution_records)
    structured_records = segment_structured_records(doc)
    review_bundles, diagnostics = build_dossier_review_bundles(
        structured_records,
        entity_records,
        reference_candidates,
    )
    if run_llm:
        review_bundles = run_dossier_llm_passes(
            review_bundles,
            model=llm_model,
            max_records=max_llm_records,
            timeout_seconds=llm_timeout_seconds,
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem.replace(" ", "-").lower()
    json_path = output_root / f"{stem}-dossier-review.json"
    report_path = output_root / f"{stem}-dossier-review.txt"
    llm_report_path = output_root / f"{stem}-dossier-review-llm.txt"

    artifact = {
        "document_path": str(source_path),
        "diagnostics": to_llm_safe_jsonable(diagnostics),
        "structured_records": to_llm_safe_jsonable(structured_records),
        "review_bundles": to_llm_safe_jsonable(review_bundles),
    }
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        render_dossier_review_report(diagnostics, review_bundles, max_records=max_report_records),
        encoding="utf-8",
    )
    llm_report_path.write_text(
        render_dossier_llm_report(diagnostics, review_bundles, max_records=max_report_records),
        encoding="utf-8",
    )
    return json_path, report_path, llm_report_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the dossier review experiment.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run the deterministic dossier-review scaffold on one .txt file."
    )
    parser.add_argument("input_path", help="Path to one .txt note file.")
    parser.add_argument(
        "--output-dir",
        default="logs/dossier-review",
        help="Directory for JSON and text artifacts. Default: logs/dossier-review",
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
        help="Run the first constrained LLM pass for dossier entries.",
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
        help="Optional cap on dossier entries sent to the LLM.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request timeout for --run-llm. Default: 60",
    )
    return parser
