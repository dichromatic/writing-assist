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


class StructuredTaggedExtractionItem(BaseModel):
    """One tagged extraction item for structured reference records."""

    model_config = ConfigDict(extra="allow")

    type_tag: Literal[
        "entity_mention",
        "fact_about",
        "relationship_between",
        "event_description",
        "unclassified",
    ]
    subject_names: list[str] = Field(default_factory=list)
    content: str = ""
    evidence_quote: str = ""

    @field_validator("content", "evidence_quote")
    @classmethod
    def _require_non_empty_text(cls, value: str) -> str:
        """Reject degenerate tagged extraction items with empty core fields."""
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text


class StructuredTaggedExtractionResponse(BaseModel):
    """Typed response model for structured tagged extraction tasks."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    record_family: str | None = Field(
        default=None,
        validation_alias=AliasChoices("record_family", "record_type"),
    )
    extraction_items: list[StructuredTaggedExtractionItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("extraction_items", "items"),
    )


class ManuscriptEntityProfileResponse(BaseModel):
    """Typed response model for manuscript pass-1 triage tasks."""

    model_config = ConfigDict(extra="allow")

    canonical_key: str | None = None
    entity_key: str | None = None
    canonical_name: str | None = None
    source_keys: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    dominant_category: str | None = None
    category: str | None = None
    passing: bool | None = None
    failing: bool | None = None
    rationale_confidence: float | None = Field(
        default=None,
        validation_alias=AliasChoices("rationale_confidence", "confidence"),
    )
    review_required: bool | None = None
    status: str | None = None
    conflicting_categories: list[str] = Field(default_factory=list)
    uncertainty_reason: str | None = None
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

    @model_validator(mode="after")
    def _normalize_triage_semantics(self) -> "ManuscriptEntityProfileResponse":
        """Keep pass-1 semantics stable across prompt/model variants."""
        if self.failing is None and isinstance(self.review_required, bool):
            self.failing = self.review_required
        if self.passing is None and self.failing is not None:
            self.passing = not self.failing
        if self.failing is None and self.passing is not None:
            self.failing = not self.passing
        if self.review_required is None and self.failing is not None:
            self.review_required = self.failing

        if self.rationale_confidence is not None:
            self.rationale_confidence = max(0.0, min(1.0, float(self.rationale_confidence)))
        return self


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
    rationale_confidence: float | None = Field(
        default=None,
        validation_alias=AliasChoices("rationale_confidence", "confidence"),
    )
    passing: bool | None = None
    failing: bool | None = None
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
            has_resolution_signal = bool(
                (self.resolved_category or "").strip()
                and (self.review_required is False or self.passing is True)
            )
            if has_resolution_signal:
                self.resolved = True

        if self.resolved is False:
            self.review_required = True
            self.failing = True
            self.passing = False
        else:
            if self.failing is None and self.review_required is False:
                self.failing = False
            if self.passing is None and self.failing is not None:
                self.passing = not self.failing
            if self.failing is None and self.passing is not None:
                self.failing = not self.passing
        if self.rationale_confidence is not None:
            self.rationale_confidence = max(0.0, min(1.0, float(self.rationale_confidence)))
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
    if task_family == LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION:
        return StructuredTaggedExtractionResponse
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
    fallback_canonical_key: str | None = None,
) -> NormalizedLLMResponseEnvelope:
    """Validate and normalize one raw payload into a stable envelope."""
    model_cls = _model_for_family(task_family)
    payload = dict(raw_payload)

    # Coerce common confidence-label drift into numeric values so strict
    # schema validation can remain enabled without discarding usable outputs.
    confidence_label_map = {
        "low": 0.30,
        "moderate": 0.60,
        "medium": 0.60,
        "high": 0.85,
    }
    for field in ("rationale_confidence", "confidence"):
        value = payload.get(field)
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapped = confidence_label_map.get(normalized)
            if mapped is None:
                for label, score in confidence_label_map.items():
                    if normalized.startswith(label):
                        mapped = score
                        break
            if mapped is not None:
                payload[field] = mapped

    # Backfill missing canonical key from deterministic packet source id.
    if fallback_canonical_key:
        if task_family in {
            LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE,
            LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION,
        }:
            if not str(payload.get("canonical_key", "")).strip():
                payload["canonical_key"] = fallback_canonical_key

    # Recover known envelope drifts before model validation.
    if task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE:
        if isinstance(payload.get("entity_profile"), dict):
            candidate = dict(payload["entity_profile"])
            for key in ("canonical_key", "source_keys", "aliases", "dominant_category", "review_required"):
                if key not in payload and key in candidate:
                    payload[key] = candidate[key]
            if "evidence" not in payload and isinstance(candidate.get("evidence"), list):
                payload["evidence"] = candidate["evidence"]
    if task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION:
        # Recover common field-name drift from pass-2 prompts.
        if "resolved_category" not in payload and isinstance(payload.get("category"), str):
            payload["resolved_category"] = payload.get("category")
        if "resolved_canonical_name" not in payload:
            identity = payload.get("identity")
            if not isinstance(identity, str):
                identity = payload.get("resolved_identity")
            if isinstance(identity, str):
                payload["resolved_canonical_name"] = identity
        if "resolution_rationale" not in payload and isinstance(payload.get("rationale"), str):
            payload["resolution_rationale"] = payload.get("rationale")

        # Backfill triage booleans from available pass-2 signals and coerce
        # malformed non-bool outputs.
        for key in ("passing", "failing", "review_required"):
            if isinstance(payload.get(key), bool):
                continue
            payload.pop(key, None)
        if "review_required" not in payload:
            payload["review_required"] = False if payload.get("resolved_category") else True
        if "passing" not in payload:
            payload["passing"] = bool(payload.get("resolved_category")) and payload["review_required"] is False
        if "failing" not in payload:
            payload["failing"] = not bool(payload.get("passing"))

    try:
        validated = model_cls.model_validate(payload)
    except ValidationError as exc:
        return NormalizedLLMResponseEnvelope(
            is_valid=False,
            proposal_payload={},
            validation_errors=[str(err) for err in exc.errors()],
            raw_payload=payload,
        )
    proposal_payload = validated.model_dump(mode="json", exclude_none=True)
    completeness_errors: list[str] = []
    if task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE:
        has_key = bool(
            str(proposal_payload.get("canonical_key", "")).strip()
            or str(proposal_payload.get("entity_key", "")).strip()
        )
        has_category = bool(
            str(proposal_payload.get("dominant_category", "")).strip()
            or str(proposal_payload.get("category", "")).strip()
        )
        has_triage_flags = all(
            isinstance(proposal_payload.get(field), bool)
            for field in ("passing", "failing", "review_required")
        )
        if not has_key:
            completeness_errors.append("missing required triage key: canonical_key or entity_key")
        if not has_category:
            completeness_errors.append("missing required triage category: dominant_category or category")
        if not has_triage_flags:
            completeness_errors.append("missing required triage booleans: passing, failing, review_required")
    if task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION:
        has_key = bool(str(proposal_payload.get("canonical_key", "")).strip())
        has_resolution_flags = all(
            isinstance(proposal_payload.get(field), bool)
            for field in ("passing", "failing", "review_required")
        )
        has_resolution_signal = any(
            bool(str(proposal_payload.get(field, "")).strip())
            for field in ("resolved_category", "remaining_uncertainty", "resolution_rationale")
        )
        if not has_key:
            completeness_errors.append("missing required resolution key: canonical_key")
        if not has_resolution_flags:
            completeness_errors.append("missing required resolution booleans: passing, failing, review_required")
        if not has_resolution_signal:
            completeness_errors.append(
                "missing required resolution signal: one of resolved_category, remaining_uncertainty, resolution_rationale"
            )
    if task_family == LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION:
        items = proposal_payload.get("extraction_items")
        if not isinstance(items, list):
            completeness_errors.append("missing required extraction_items list")
        else:
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    completeness_errors.append(f"extraction_items[{index}] is not an object")
                    continue
                if not str(item.get("content", "")).strip():
                    completeness_errors.append(f"extraction_items[{index}] missing non-empty content")
                if not str(item.get("evidence_quote", "")).strip():
                    completeness_errors.append(f"extraction_items[{index}] missing non-empty evidence_quote")
    if completeness_errors:
        return NormalizedLLMResponseEnvelope(
            is_valid=False,
            proposal_payload={},
            validation_errors=completeness_errors,
            raw_payload=payload,
        )

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
        raw_payload=payload,
    )
