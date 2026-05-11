"""Compatibility imports for the former dossier-review experiment package."""

from backend.nlp.experiments.structured_review import (
    build_dossier_review_bundles,
    build_structured_review_bundles,
    render_dossier_review_report,
    render_structured_review_report,
    run_dossier_llm_pass,
    run_dossier_llm_passes,
    run_structured_llm_pass,
    run_structured_llm_passes,
)

__all__ = [
    "build_dossier_review_bundles",
    "build_structured_review_bundles",
    "render_dossier_review_report",
    "render_structured_review_report",
    "run_dossier_llm_pass",
    "run_dossier_llm_passes",
    "run_structured_llm_pass",
    "run_structured_llm_passes",
]
