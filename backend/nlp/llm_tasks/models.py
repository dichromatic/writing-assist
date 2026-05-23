"""
LLM response model and normalization for rescue verification tasks.

.. code-block:: mermaid

    flowchart TD
        A[Raw parsed JSON from LLM] --> B[Confidence label coercion]
        B --> C[Pydantic model validation]
        C -->|valid| D[Completeness checks]
        C -->|invalid| E[Validation error envelope]
        D -->|pass| F[Normalized proposal envelope]
        D -->|fail| E
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SuppressionRescueResponse(BaseModel):
    """Typed response model for suppression rescue verification tasks.

    Args:
        normalized_key: Entity identifier from the deterministic pipeline.
        rescue: Whether this entity should be promoted past suppression.
        entity_type: Category when rescued (character/place/group/object/event/concept).
        canonical_name: Best display name for this entity.
        confidence: LLM self-reported confidence in the verdict.
        rationale: One sentence explaining the rescue or suppression verdict.
    """

    model_config = ConfigDict(extra="allow")

    normalized_key: str | None = None
    rescue: bool = False
    entity_type: str | None = None
    canonical_name: str | None = None
    confidence: float | None = None
    rationale: str | None = None


class NormalizedResponseEnvelope(BaseModel):
    """Serializable envelope persisted in LLMTaskResult.payload.

    Args:
        schema_version: Envelope schema version for forward compatibility.
        is_valid: Whether the response passed validation and completeness.
        proposal_payload: Validated and normalized response fields.
        validation_errors: Reasons for rejection when is_valid is false.
        raw_payload: Original parsed JSON for debugging rejected responses.
    """

    schema_version: Literal["1"] = "1"
    is_valid: bool
    proposal_payload: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


_CONFIDENCE_LABEL_MAP = {
    "low": 0.30,
    "moderate": 0.60,
    "medium": 0.60,
    "high": 0.85,
}


def normalize_rescue_payload(
    raw_payload: dict[str, Any],
    *,
    fallback_normalized_key: str | None = None,
) -> NormalizedResponseEnvelope:
    """Validate and normalize one raw LLM response into a stable envelope.

    Args:
        raw_payload: Parsed JSON from the LLM response body.
        fallback_normalized_key: Entity key from the task packet, used
            when the LLM omits normalized_key from its response.

    Returns:
        Envelope with validated proposal or validation errors.
    """
    payload = dict(raw_payload)

    # Coerce string confidence labels into numeric values.
    value = payload.get("confidence")
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapped = _CONFIDENCE_LABEL_MAP.get(normalized)
        if mapped is None:
            for label, score in _CONFIDENCE_LABEL_MAP.items():
                if normalized.startswith(label):
                    mapped = score
                    break
        if mapped is not None:
            payload["confidence"] = mapped

    # Backfill normalized_key from the deterministic packet source id.
    if fallback_normalized_key:
        if not str(payload.get("normalized_key", "")).strip():
            payload["normalized_key"] = fallback_normalized_key

    try:
        validated = SuppressionRescueResponse.model_validate(payload)
    except ValidationError as exc:
        return NormalizedResponseEnvelope(
            is_valid=False,
            proposal_payload={},
            validation_errors=[str(err) for err in exc.errors()],
            raw_payload=payload,
        )

    proposal_payload = validated.model_dump(mode="json", exclude_none=True)

    completeness_errors: list[str] = []
    if not isinstance(proposal_payload.get("rescue"), bool):
        completeness_errors.append("missing required boolean: rescue")
    if proposal_payload.get("rescue") is True:
        if not str(proposal_payload.get("entity_type", "")).strip():
            completeness_errors.append("rescue=true requires non-empty entity_type")
    if completeness_errors:
        return NormalizedResponseEnvelope(
            is_valid=False,
            proposal_payload={},
            validation_errors=completeness_errors,
            raw_payload=payload,
        )

    return NormalizedResponseEnvelope(
        is_valid=True,
        proposal_payload=proposal_payload,
        validation_errors=[],
        raw_payload=payload,
    )
