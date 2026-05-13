"""Shared LLM task schema identifiers and lookup helpers."""

from __future__ import annotations

from backend.nlp.types import LLMTaskFamily

TASK_SCHEMA_IDS: dict[LLMTaskFamily, str] = {
    LLMTaskFamily.RECORD_FACT_EXTRACTION: "record_fact_extraction.v1",
    LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE: "manuscript_entity_profile.v1",
    LLMTaskFamily.MANUSCRIPT_REFERENCE_ATTACHMENT: "manuscript_reference_attachment.v1",
    LLMTaskFamily.MANUSCRIPT_CATEGORY_RESOLUTION: "manuscript_category_resolution.v1",
    LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION: "manuscript_entity_review_resolution.v1",
}


def schema_id_for(task_family: LLMTaskFamily) -> str:
    """Return the schema identifier for one task family."""
    return TASK_SCHEMA_IDS[task_family]
