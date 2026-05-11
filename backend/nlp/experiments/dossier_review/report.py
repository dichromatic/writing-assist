"""Compatibility wrapper for the former dossier-review report module."""

from backend.nlp.experiments.structured_review.report import (
    render_dossier_llm_report,
    render_dossier_review_report,
    render_structured_llm_report,
    render_structured_review_report,
)

__all__ = [
    "render_dossier_llm_report",
    "render_dossier_review_report",
    "render_structured_llm_report",
    "render_structured_review_report",
]
