"""Compatibility wrapper for the former dossier-review prompt module."""

from backend.nlp.experiments.structured_review.prompt_packet import (
    build_dossier_prompt_packet,
    build_record_prompt_packet,
)

__all__ = [
    "build_dossier_prompt_packet",
    "build_record_prompt_packet",
]
