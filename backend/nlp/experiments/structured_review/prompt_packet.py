"""
Structured-record prompt packet builder - freeze deterministic evidence for LLM handoff.

.. code-block:: mermaid

    flowchart TD
        A[StructuredRecord] --> B[Record metadata]
        C[DeterministicSeedBundle] --> D[Seed evidence]
        E[DeterministicFactCandidate list] --> F[Fact candidates]
        B & D & F --> G[Dispatch task by record type]
        G --> H[LLMRecordPromptPacket]
"""

from __future__ import annotations

from backend.nlp.text_filtering import sanitize_for_llm
from backend.nlp.types import (
    DeterministicFactCandidate,
    DeterministicSeedBundle,
    LLMRecordPromptPacket,
    StructuredRecord,
    StructuredRecordType,
)

_DOSSIER_TASK_CONSTRAINTS = [
    "Identify the dossier subject only when the entry text supports it.",
    "Extract only explicit subject facts stated in the entry.",
    "Do not infer personality, motivations, or unstated relationships.",
    "Do not write canon updates directly; return proposals only in later phases.",
    "Preserve ambiguity as open questions when the subject or fact is unclear.",
]

_REFERENCE_SECTION_TASK_CONSTRAINTS = [
    "Extract only explicit section facts and named concepts stated in the record.",
    "Preserve explanatory prose as facts only when it is directly stated.",
    "Do not infer canon updates or unstated relationships.",
    "Do not write canon updates directly; return proposals only in later phases.",
    "Preserve ambiguity as open questions when structure or scope is unclear.",
]

_OUTLINE_BEAT_TASK_CONSTRAINTS = [
    "Extract only explicit beat facts, participants, and planned actions stated in the record.",
    "Treat outline bullets as planning statements, not final canon truth.",
    "Do not infer unstated motivations or consequences.",
    "Do not write canon updates directly; return proposals only in later phases.",
    "Preserve ambiguity as open questions when the beat scope is unclear.",
]

_LOOSE_RECORD_TASK_CONSTRAINTS = [
    "Extract only explicit note facts or useful context stated in the loose record.",
    "Treat loose records as low-structure evidence, not as finalized canon.",
    "Do not invent a subject when the record does not clearly support one.",
    "Do not write canon updates directly; return proposals only in later phases.",
    "Preserve ambiguity as open questions when the loose record scope is unclear.",
]


def _source_authority(record: StructuredRecord) -> str:
    """Return the document-family authority label for one structured record.

    Args:
        record: Structured record being prepared for later LLM review.

    Returns:
        Source authority label tied to path and record family.
    """
    lower_path = record.document_path.casefold()
    if record.record_type == StructuredRecordType.DOSSIER_ENTRY:
        return "planning_dossier"
    if record.record_type == StructuredRecordType.OUTLINE_BEAT:
        return "planning_outline"
    if "world context" in lower_path:
        return "world_context_reference"
    if "story planning" in lower_path:
        return "planning_reference_section"
    return "structured_note_reference"


def _task_fields(record: StructuredRecord) -> tuple[str, str, list[str]]:
    """Return task metadata for one supported structured record family.

    Args:
        record: Structured record being prepared for later LLM review.

    Returns:
        Task name, task goal, and task constraints.
    """
    if record.record_type == StructuredRecordType.DOSSIER_ENTRY:
        return (
            "dossier_subject_and_explicit_facts",
            "Identify the dossier subject and extract explicit subject facts only from this record.",
            list(_DOSSIER_TASK_CONSTRAINTS),
        )
    if record.record_type == StructuredRecordType.OUTLINE_BEAT:
        return (
            "outline_beat_explicit_facts",
            "Extract explicit beat facts, participants, and planned actions only from this record.",
            list(_OUTLINE_BEAT_TASK_CONSTRAINTS),
        )
    if record.record_type == StructuredRecordType.LOOSE_RECORD:
        return (
            "loose_record_explicit_context",
            "Extract explicit useful facts or context only from this low-structure record.",
            list(_LOOSE_RECORD_TASK_CONSTRAINTS),
        )
    return (
        "reference_section_explicit_facts",
        "Extract explicit section facts, named concepts, places, groups, and relationship-like statements only from this record.",
        list(_REFERENCE_SECTION_TASK_CONSTRAINTS),
    )


def build_record_prompt_packet(
    record: StructuredRecord,
    seed_bundle: DeterministicSeedBundle,
    fact_candidates: list[DeterministicFactCandidate],
) -> LLMRecordPromptPacket:
    """Build the future LLM input packet for one supported record.

    Args:
        record: Structured record being reviewed.
        seed_bundle: Deterministic seed evidence for the same record.
        fact_candidates: Shallow deterministic fact-like candidates preserved
            before any model call.

    Returns:
        Structured packet that freezes the task boundary and deterministic
        evidence for later model-assisted interpretation.
    """
    sanitized_seed_bundle = sanitize_for_llm(seed_bundle)
    sanitized_fact_candidates = sanitize_for_llm(fact_candidates)

    task_name, task_goal, task_constraints = _task_fields(record)

    return LLMRecordPromptPacket(
        record_id=record.record_id,
        record_type=record.record_type,
        document_path=record.document_path,
        source_authority=_source_authority(record),
        task_name=task_name,
        task_goal=task_goal,
        task_constraints=task_constraints,
        raw_record_text=sanitize_for_llm(record.raw_text),
        header_line=sanitize_for_llm(record.heading_text),
        parent_heading=sanitize_for_llm(record.parent_heading),
        deterministic_seed_bundle=sanitized_seed_bundle,
        deterministic_fact_candidates=sanitized_fact_candidates,
    )
