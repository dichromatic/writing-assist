"""
Typed LLM response models and normalization for shared task families.

.. code-block:: mermaid

    flowchart TD
        A[Raw parsed JSON payload] --> B[Task-family schema selection]
        B --> C[Pydantic model validation]
        C -->|valid| D[Normalized proposal payload]
        C -->|invalid| E[Validation error diagnostics]
        D & E --> F[LLM result payload envelope]
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from backend.nlp.types import LLMTaskFamily


class LLMEvidencePayload(BaseModel):
    """Evidence item model used by manuscript task-family responses."""

    model_config = ConfigDict(extra="allow")

    evidence_id: str
    document_path: str = Field(
        default="",
        validation_alias=AliasChoices("document_path", "document"),
    )
    quote: str
    confidence_score: float | None = Field(
        default=None,
        validation_alias=AliasChoices("confidence_score", "confidence"),
    )
    visibility_bucket: str | None = None
    suppression_reason: str | None = None


class RecordFactExtractionFact(BaseModel):
    """One extracted record fact statement and attached evidence IDs."""

    model_config = ConfigDict(extra="allow")

    statement: str = Field(
        default="",
        validation_alias=AliasChoices("statement", "proposition", "text"),
    )
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _synthesize_statement_from_triplet(self) -> "RecordFactExtractionFact":
        """Fallback when the model returns subject/predicate/object triplets."""
        if self.statement.strip():
            return self
        parts = [part for part in (self.subject, self.predicate, self.object) if part]
        if parts:
            self.statement = " ".join(parts)
        return self


class RecordFactExtractionResponse(BaseModel):
    """Typed response model for record fact extraction tasks."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    record_family: str | None = Field(
        default=None,
        validation_alias=AliasChoices("record_family", "record_type"),
    )
    source_document_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_document_path", "source_document"),
    )
    facts: list[RecordFactExtractionFact] = Field(
        default_factory=list,
        validation_alias=AliasChoices("facts", "propositions"),
    )


class ManuscriptEntityProfileResponse(BaseModel):
    """Typed response model for manuscript entity-profile tasks."""

    model_config = ConfigDict(extra="allow")

    canonical_key: str | None = None
    entity_key: str | None = None
    canonical_name: str | None = None
    source_keys: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    dominant_category: str | None = None
    category: str | None = None
    review_required: bool | None = None
    status: str | None = None
    conflicting_categories: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    notes: str | None = None
    evidence: list[LLMEvidencePayload] = Field(default_factory=list)

    @field_validator(
        "source_keys",
        "aliases",
        "conflicting_categories",
        "reasons",
        "conflicts",
        mode="before",
    )
    @classmethod
    def _coerce_nullable_lists(cls, value):
        """Accept null list fields from model output as empty lists."""
        if value is None:
            return []
        return value


class ManuscriptReferenceAttachmentCandidate(BaseModel):
    """One attachment candidate returned by the model."""

    target_entity_key: str | None = None
    target_entity: str | None = None
    reference_span_text: str | None = None
    confidence: float | None = None
    reason: str | None = None


class ManuscriptReferenceAttachmentResponse(BaseModel):
    """Typed response model for manuscript reference-attachment tasks."""

    model_config = ConfigDict(extra="allow")

    reference_id: str | None = None
    reference_text: str | None = None
    candidates: list[ManuscriptReferenceAttachmentCandidate] = Field(default_factory=list)
    selected_target_entity_key: str | None = None
    selected_target_entity: str | None = None
    unresolved: bool | None = None
    review_required: bool | None = None
    rationale: str | None = None


class ManuscriptCategoryResolutionResponse(BaseModel):
    """Typed response model for manuscript category-resolution tasks."""

    model_config = ConfigDict(extra="allow")

    canonical_key: str | None = None
    category: str | None = None
    confidence: float | None = None
    review_required: bool | None = None
    rationale: str | None = None
    alternatives: list[str] = Field(default_factory=list)


class ManuscriptEntityReviewResolutionResponse(BaseModel):
    """Typed response model for second-pass manuscript review resolution."""

    model_config = ConfigDict(extra="allow")

    canonical_key: str | None = None
    resolved: bool = False
    resolved_category: str | None = None
    resolved_canonical_name: str | None = None
    resolved_aliases: list[str] = Field(default_factory=list)
    resolution_confidence: float | None = None
    resolution_rationale: str | None = None
    remaining_uncertainty: str | None = None
    review_required: bool = True
    evidence: list[LLMEvidencePayload] = Field(default_factory=list)
    synthesis_notes: str | None = None

    @model_validator(mode="after")
    def _enforce_resolution_review_consistency(self) -> "ManuscriptEntityReviewResolutionResponse":
        """Force unresolved outcomes to remain review-required.

        A second-pass response that leaves an entity unresolved must keep
        review_required true so downstream workflows do not treat it as closed.
        """
        if self.resolved is False:
            self.review_required = True
        return self


class NormalizedLLMResponseEnvelope(BaseModel):
    """Serializable envelope persisted in LLMTaskResult.payload."""

    schema_version: Literal["1"] = "1"
    is_valid: bool
    proposal_payload: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


def _model_for_family(task_family: LLMTaskFamily) -> type[BaseModel]:
    """Return the typed response model expected for one task family."""
    if task_family == LLMTaskFamily.RECORD_FACT_EXTRACTION:
        return RecordFactExtractionResponse
    if task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE:
        return ManuscriptEntityProfileResponse
    if task_family == LLMTaskFamily.MANUSCRIPT_REFERENCE_ATTACHMENT:
        return ManuscriptReferenceAttachmentResponse
    if task_family == LLMTaskFamily.MANUSCRIPT_CATEGORY_RESOLUTION:
        return ManuscriptCategoryResolutionResponse
    if task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION:
        return ManuscriptEntityReviewResolutionResponse
    raise ValueError(f"Unsupported task_family for validation: {task_family.value}")


def normalize_llm_payload(
    *,
    task_family: LLMTaskFamily,
    raw_payload: dict[str, Any],
) -> NormalizedLLMResponseEnvelope:
    """Validate and normalize one raw payload into a stable envelope."""
    model_cls = _model_for_family(task_family)
    # Recover known envelope drifts before model validation.
    if task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE:
        if isinstance(raw_payload.get("entity_profile"), dict):
            candidate = dict(raw_payload["entity_profile"])
            for key in ("canonical_key", "source_keys", "aliases", "dominant_category", "review_required"):
                if key not in raw_payload and key in candidate:
                    raw_payload[key] = candidate[key]
            if "evidence" not in raw_payload and isinstance(candidate.get("evidence"), list):
                raw_payload["evidence"] = candidate["evidence"]

    try:
        validated = model_cls.model_validate(raw_payload)
    except ValidationError as exc:
        return NormalizedLLMResponseEnvelope(
            is_valid=False,
            proposal_payload={},
            validation_errors=[str(err) for err in exc.errors()],
            raw_payload=raw_payload,
        )
    proposal_payload = validated.model_dump(mode="json", exclude_none=True)
    if task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION:
        unresolved = bool(proposal_payload.get("resolved") is False)
        has_category = bool(str(proposal_payload.get("resolved_category", "")).strip())
        has_uncertainty = bool(str(proposal_payload.get("remaining_uncertainty", "")).strip())
        rationale_text = (
            str(proposal_payload.get("resolution_rationale", "")).strip()
            or str(proposal_payload.get("justification", "")).strip()
            or str(proposal_payload.get("resolution_notes", "")).strip()
            or str(proposal_payload.get("notes", "")).strip()
        )
        # Conservative promotion signal for unresolved-but-strongly-argued
        # outputs. This does not auto-resolve; it flags a candidate for the
        # next decision layer.
        if unresolved and has_category and rationale_text and not has_uncertainty:
            proposal_payload["resolution_candidate"] = True
            proposal_payload["candidate_reason"] = rationale_text
        else:
            proposal_payload["resolution_candidate"] = False

    return NormalizedLLMResponseEnvelope(
        is_valid=True,
        proposal_payload=proposal_payload,
        validation_errors=[],
        raw_payload=raw_payload,
    )
