"""Compatibility wrapper for the former dossier-review CLI module."""

from backend.nlp.experiments.structured_review.cli import (
    build_parser,
    run_dossier_review_experiment,
    run_structured_review_experiment,
)

__all__ = [
    "build_parser",
    "run_dossier_review_experiment",
    "run_structured_review_experiment",
]
