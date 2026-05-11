"""Compatibility wrapper for the former dossier-review LLM module."""

from backend.nlp.experiments.structured_review.llm_pass import (
    _extract_json_object_text,
    run_dossier_llm_pass,
    run_dossier_llm_passes,
    run_structured_llm_pass,
    run_structured_llm_passes,
)

__all__ = [
    "_extract_json_object_text",
    "run_dossier_llm_pass",
    "run_dossier_llm_passes",
    "run_structured_llm_pass",
    "run_structured_llm_passes",
]
