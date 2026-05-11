"""Compatibility wrapper for the former dossier-review bundle module."""

from backend.nlp.experiments.structured_review.review_bundle import (
    build_dossier_review_bundles,
    build_structured_review_bundles,
)

__all__ = [
    "build_dossier_review_bundles",
    "build_structured_review_bundles",
]
