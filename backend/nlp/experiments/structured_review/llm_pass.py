"""
Structured review LLM pass - run constrained model extraction over one bundle.

.. code-block:: mermaid

    flowchart TD
        A[RecordReviewBundle with LLMRecordPromptPacket] --> B[Build structured prompt]
        B --> C{Provider wire format}
        C -->|Responses API| D[Call OpenAI Responses endpoint]
        C -->|Chat Completions| E[Call OpenAI compatible chat endpoint]
        D --> F[Parse structured JSON output]
        E --> F
        F --> G[Fill llm_subject_proposal and llm_fact_proposals]
        G --> H[Compute deterministic vs LLM comparison fields]
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable
from urllib import error, request

from backend.nlp.text_filtering import sanitize_for_llm
from backend.nlp.types import PendingLLMResponse, RecordReviewBundle

_DEFAULT_BASE_URL = "https://api.openai.com/v1/responses"
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _response_schema() -> dict[str, Any]:
    """Return the structured output schema for the first structured LLM pass.

    Returns:
        JSON Schema object for constrained subject and fact extraction.
    """
    return {
        "name": "structured_record_subject_and_explicit_facts",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subject": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "subject_name": {"type": "string"},
                        "alternate_names": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence_quotes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "certainty_note": {"type": "string"},
                        "unresolved": {"type": "boolean"},
                    },
                    "required": [
                        "subject_name",
                        "alternate_names",
                        "evidence_quotes",
                        "certainty_note",
                        "unresolved",
                    ],
                },
                "fact_proposals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                            "evidence_quote": {"type": "string"},
                            "certainty_note": {"type": "string"},
                        },
                        "required": [
                            "label",
                            "value",
                            "evidence_quote",
                            "certainty_note",
                        ],
                    },
                },
                "open_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["subject", "fact_proposals", "open_questions"],
        },
    }


def _build_request_payload(bundle: RecordReviewBundle, model: str) -> dict[str, Any]:
    """Build the Responses API request body for one review bundle.

    Args:
        bundle: Review bundle carrying the prompt packet.
        model: Model name to use for the call.

    Returns:
        JSON body for the OpenAI Responses API.
    """
    packet = bundle.llm_prompt_packet
    prompt_json = json.dumps({
        "record_id": packet.record_id,
        "record_type": packet.record_type.value,
        "document_path": packet.document_path,
        "source_authority": packet.source_authority,
        "task_name": packet.task_name,
        "task_goal": packet.task_goal,
        "task_constraints": packet.task_constraints,
        "header_line": packet.header_line,
        "parent_heading": packet.parent_heading,
        "raw_record_text": packet.raw_record_text,
        "deterministic_seed_bundle": {
            "header_line": packet.deterministic_seed_bundle.header_line,
            "suspected_subject_guess": (
                {
                    "primary_guess": packet.deterministic_seed_bundle.suspected_subject_guess.primary_guess,
                    "alternative_guesses": packet.deterministic_seed_bundle.suspected_subject_guess.alternative_guesses,
                    "reason": packet.deterministic_seed_bundle.suspected_subject_guess.reason,
                }
                if packet.deterministic_seed_bundle.suspected_subject_guess is not None
                else None
            ),
            "candidate_rank_texts": packet.deterministic_seed_bundle.candidate_rank_texts,
            "field_lines": [
                {
                    "line_type": field_line.line_type.value,
                    "raw_text": field_line.raw_text,
                    "label": field_line.label,
                    "value": field_line.value,
                }
                for field_line in packet.deterministic_seed_bundle.field_lines
            ],
            "entity_candidates": [
                {
                    "normalized_key": entity.normalized_key,
                    "surface_forms": entity.surface_forms,
                    "winning_category": entity.winning_category.value,
                    "bucket": entity.bucket.value,
                    "suppression_reason": (
                        entity.suppression_reason.value if entity.suppression_reason else ""
                    ),
                    "confidence_score": entity.confidence_score,
                    "bucket_detail": entity.bucket_detail,
                }
                for entity in packet.deterministic_seed_bundle.entity_candidates
            ],
            "reference_candidates": [
                {
                    "normalized": reference.normalized,
                    "reference_type": reference.reference_type.value,
                    "linked_entity_keys": reference.linked_entity_keys,
                    "context_before": reference.context_before,
                    "context_after": reference.context_after,
                }
                for reference in packet.deterministic_seed_bundle.reference_candidates
            ],
            "known_canon_matches": packet.deterministic_seed_bundle.known_canon_matches,
            "structural_flags": packet.deterministic_seed_bundle.structural_flags,
        },
        "deterministic_fact_candidates": [
            {
                "label": candidate.label,
                "value": candidate.value,
                "reason": candidate.reason,
            }
            for candidate in packet.deterministic_fact_candidates
        ],
    }, ensure_ascii=False, indent=2)

    system_text = "\n".join([
        "You extract subjects and explicit facts from one structured record.",
        "Return JSON only through the provided schema.",
        "Do not infer unstated personality, motivations, or relationships.",
        "If the subject is unclear, mark it unresolved and add open questions.",
        "Treat deterministic candidates as weak hints, not truth.",
    ])

    return {
        "model": model,
        "input": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt_json},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                **_response_schema(),
            }
        },
    }


def _build_chat_completions_payload(bundle: RecordReviewBundle, model: str) -> dict[str, Any]:
    """Build the chat-completions request body for one review bundle.

    Args:
        bundle: Review bundle carrying the prompt packet.
        model: Model name to use for the call.

    Returns:
        JSON body for an OpenAI-compatible chat-completions endpoint.
    """
    packet = bundle.llm_prompt_packet
    packet_json = json.dumps({
        "record_id": packet.record_id,
        "record_type": packet.record_type.value,
        "document_path": packet.document_path,
        "source_authority": packet.source_authority,
        "task_name": packet.task_name,
        "task_goal": packet.task_goal,
        "task_constraints": packet.task_constraints,
        "header_line": packet.header_line,
        "parent_heading": packet.parent_heading,
        "raw_record_text": packet.raw_record_text,
        "deterministic_seed_bundle": {
            "header_line": packet.deterministic_seed_bundle.header_line,
            "suspected_subject_guess": (
                {
                    "primary_guess": packet.deterministic_seed_bundle.suspected_subject_guess.primary_guess,
                    "alternative_guesses": packet.deterministic_seed_bundle.suspected_subject_guess.alternative_guesses,
                    "reason": packet.deterministic_seed_bundle.suspected_subject_guess.reason,
                }
                if packet.deterministic_seed_bundle.suspected_subject_guess is not None
                else None
            ),
            "candidate_rank_texts": packet.deterministic_seed_bundle.candidate_rank_texts,
            "field_lines": [
                {
                    "line_type": field_line.line_type.value,
                    "raw_text": field_line.raw_text,
                    "label": field_line.label,
                    "value": field_line.value,
                }
                for field_line in packet.deterministic_seed_bundle.field_lines
            ],
            "entity_candidates": [
                {
                    "normalized_key": entity.normalized_key,
                    "surface_forms": entity.surface_forms,
                    "winning_category": entity.winning_category.value,
                    "bucket": entity.bucket.value,
                    "suppression_reason": (
                        entity.suppression_reason.value if entity.suppression_reason else ""
                    ),
                    "confidence_score": entity.confidence_score,
                    "bucket_detail": entity.bucket_detail,
                }
                for entity in packet.deterministic_seed_bundle.entity_candidates
            ],
            "reference_candidates": [
                {
                    "normalized": reference.normalized,
                    "reference_type": reference.reference_type.value,
                    "linked_entity_keys": reference.linked_entity_keys,
                    "context_before": reference.context_before,
                    "context_after": reference.context_after,
                }
                for reference in packet.deterministic_seed_bundle.reference_candidates
            ],
            "known_canon_matches": packet.deterministic_seed_bundle.known_canon_matches,
            "structural_flags": packet.deterministic_seed_bundle.structural_flags,
        },
        "deterministic_fact_candidates": [
            {
                "label": candidate.label,
                "value": candidate.value,
                "reason": candidate.reason,
            }
            for candidate in packet.deterministic_fact_candidates
        ],
        "required_output_schema": _response_schema()["schema"],
    }, ensure_ascii=False, indent=2)

    system_text = "\n".join([
        "You extract subjects and explicit facts from one structured record.",
        "Return JSON only.",
        "The JSON must match the provided required_output_schema exactly.",
        "Do not infer unstated personality, motivations, or relationships.",
        "If the subject is unclear, mark it unresolved and add open questions.",
        "Treat deterministic candidates as weak hints, not truth.",
    ])

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": packet_json},
        ],
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stream": False,
    }


def _extract_output_text(response_body: dict[str, Any]) -> str:
    """Extract the generated text payload from a Responses API response.

    Args:
        response_body: Raw decoded JSON response.

    Returns:
        Text output that should contain the structured JSON payload.

    Raises:
        ValueError: If no text output can be located.
    """
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    for item in response_body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text_value = content.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value

    raise ValueError("Responses API did not return any structured output text")


def _extract_chat_completion_text(response_body: dict[str, Any]) -> str:
    """Extract the generated text payload from a chat-completions response.

    Args:
        response_body: Raw decoded JSON response.

    Returns:
        Text output that should contain the structured JSON payload.

    Raises:
        ValueError: If no assistant message content can be located.
    """
    choices = response_body.get("choices", [])
    if not choices or not isinstance(choices, list):
        raise ValueError("Chat completions response did not include choices")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content

    raise ValueError("Chat completions response did not include assistant content")


def _extract_json_object_text(text: str) -> str:
    """Extract the first top-level JSON object from model text.

    Args:
        text: Raw model output text, possibly with fences or preamble.

    Returns:
        A substring that should decode as one JSON object.

    Raises:
        ValueError: If no balanced top-level JSON object can be found.
    """
    fence_stripped = text.strip()
    if fence_stripped.startswith("```"):
        lines = fence_stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        fence_stripped = "\n".join(lines).strip()

    start_index = fence_stripped.find("{")
    if start_index == -1:
        raise ValueError("Model output did not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(fence_stripped)):
        char = fence_stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return fence_stripped[start_index:index + 1]

    raise ValueError("Model output contained an unbalanced JSON object")


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve the live provider API key from explicit or environment values.

    Args:
        api_key: Explicit API key override.

    Returns:
        Resolved bearer token.

    Raises:
        RuntimeError: If no supported key is present.
    """
    resolved_api_key = (
        api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("NIM_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
    )
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY, NIM_API_KEY, or NVIDIA_API_KEY is not set")
    return resolved_api_key


def _resolve_base_url(explicit_base_url: str | None) -> str:
    """Resolve the live provider base URL from explicit or environment values.

    Args:
        explicit_base_url: Explicit endpoint override.

    Returns:
        Resolved provider endpoint or base URL.
    """
    return (
        explicit_base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("NIM_BASE_URL")
        or os.getenv("NVIDIA_BASE_URL")
        or _DEFAULT_BASE_URL
    )


def _is_chat_completions_style(base_url: str) -> bool:
    """Return whether the configured provider endpoint expects chat completions.

    Args:
        base_url: Configured endpoint or provider base URL.

    Returns:
        True when the live path should use chat completions.
    """
    normalized = base_url.rstrip("/")
    return (
        normalized == _DEFAULT_NVIDIA_BASE_URL
        or normalized.endswith("/chat/completions")
    )


def _call_openai_responses_api(
    bundle: RecordReviewBundle,
    *,
    model: str,
    api_key: str | None,
    base_url: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    """Call the OpenAI Responses API for one review bundle.

    Args:
        bundle: Review bundle carrying the prompt packet.
        model: Model name to use for the request.
        api_key: Explicit API key, or None to read from the environment.
        base_url: Responses API endpoint.
        timeout_seconds: HTTP timeout.

    Returns:
        Parsed JSON body and provider response id.
    """
    resolved_api_key = _resolve_api_key(api_key)

    payload = _build_request_payload(bundle, model)
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        base_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc

    return response_body, str(response_body.get("id", ""))


def _call_openai_chat_completions_api(
    bundle: RecordReviewBundle,
    *,
    model: str,
    api_key: str | None,
    base_url: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    """Call an OpenAI-compatible chat-completions endpoint for one bundle.

    Args:
        bundle: Review bundle carrying the prompt packet.
        model: Model name to use for the request.
        api_key: Explicit API key, or None to read from the environment.
        base_url: Provider base URL or direct chat endpoint.
        timeout_seconds: HTTP timeout.

    Returns:
        Parsed structured JSON body and provider response id.
    """
    resolved_api_key = _resolve_api_key(api_key)
    endpoint = (
        base_url.rstrip("/")
        if base_url.rstrip("/").endswith("/chat/completions")
        else base_url.rstrip("/") + "/chat/completions"
    )

    payload = _build_chat_completions_payload(bundle, model)
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat completions HTTP {exc.code}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Chat completions request failed: {exc.reason}") from exc

    content_text = _extract_chat_completion_text(response_body)
    parsed_output = json.loads(_extract_json_object_text(content_text))
    return parsed_output, str(response_body.get("id", ""))


def _normalize_fact_pair(label: str, value: str) -> tuple[str, str]:
    """Normalize a fact pair for deterministic side-by-side comparison."""
    return (label.strip().casefold(), value.strip().casefold())


def _apply_llm_result(
    bundle: RecordReviewBundle,
    llm_result: dict[str, Any],
    *,
    model: str,
    response_id: str,
) -> RecordReviewBundle:
    """Apply one completed structured LLM result onto a review bundle.

    Args:
        bundle: Original deterministic review bundle.
        llm_result: Parsed structured JSON produced by the model.
        model: Model name used for the call.
        response_id: Provider response identifier.

    Returns:
        New review bundle with completed LLM fields and comparison results.
    """
    updated = deepcopy(bundle)
    sanitized_result = sanitize_for_llm(llm_result)
    subject = sanitized_result["subject"]
    facts = sanitized_result["fact_proposals"]
    open_questions = sanitized_result["open_questions"]

    updated.llm_subject_proposal = PendingLLMResponse(
        status="completed",
        payload=subject,
        model=model,
        response_id=response_id,
    )
    updated.llm_fact_proposals = PendingLLMResponse(
        status="completed",
        payload={"items": facts},
        model=model,
        response_id=response_id,
    )
    updated.open_questions = list(open_questions)

    agreement_items: list[str] = []
    deterministic_only_items: list[str] = []
    llm_only_items: list[str] = []

    deterministic_subject = (
        updated.deterministic_subject_guess.primary_guess.casefold().strip()
        if updated.deterministic_subject_guess is not None
        else ""
    )
    llm_subject = str(subject.get("subject_name", "")).casefold().strip()
    if deterministic_subject and llm_subject and deterministic_subject == llm_subject:
        agreement_items.append(f"subject:{subject['subject_name']}")
    elif deterministic_subject:
        deterministic_only_items.append(
            f"subject:{updated.deterministic_subject_guess.primary_guess}"
        )
    if llm_subject and deterministic_subject != llm_subject:
        llm_only_items.append(f"subject:{subject['subject_name']}")

    deterministic_facts = {
        _normalize_fact_pair(candidate.label, candidate.value): f"{candidate.label}: {candidate.value}"
        for candidate in updated.deterministic_fact_candidates
    }
    llm_facts = {
        _normalize_fact_pair(str(item["label"]), str(item["value"])): f"{item['label']}: {item['value']}"
        for item in facts
    }

    for key, display in deterministic_facts.items():
        if key in llm_facts:
            agreement_items.append(display)
        else:
            deterministic_only_items.append(display)
    for key, display in llm_facts.items():
        if key not in deterministic_facts:
            llm_only_items.append(display)

    updated.agreement_items = agreement_items
    updated.deterministic_only_items = deterministic_only_items
    updated.llm_only_items = llm_only_items
    return updated


def _failed_bundle(
    bundle: RecordReviewBundle,
    *,
    model: str,
    error_message: str,
) -> RecordReviewBundle:
    """Return a bundle that records a failed LLM call without crashing.

    Args:
        bundle: Original deterministic bundle.
        model: Attempted model name.
        error_message: Failure detail to preserve in artifacts.

    Returns:
        New bundle with failed LLM status fields.
    """
    updated = deepcopy(bundle)
    updated.llm_subject_proposal = PendingLLMResponse(
        status="failed",
        error=error_message,
        model=model,
    )
    updated.llm_fact_proposals = PendingLLMResponse(
        status="failed",
        error=error_message,
        model=model,
    )
    updated.open_questions = [error_message]
    return updated


def run_structured_llm_pass(
    bundle: RecordReviewBundle,
    *,
    model: str = _DEFAULT_MODEL,
    responder: Callable[[RecordReviewBundle, str], tuple[dict[str, Any], str]] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 60.0,
) -> RecordReviewBundle:
    """Run the first constrained structured LLM pass for one review bundle.

    Args:
        bundle: Deterministic structured review bundle.
        model: Model name to use.
        responder: Optional injected responder for tests or alternate
            execution paths. It must return a structured result dict plus a
            provider response id.
        api_key: Optional explicit API key.
        base_url: Responses API endpoint. When omitted, the value falls back
            to OPENAI_BASE_URL or the default OpenAI endpoint.
        timeout_seconds: HTTP timeout for live calls.

    Returns:
        Updated bundle with completed or failed LLM fields.
    """
    try:
        if responder is not None:
            llm_result, response_id = responder(bundle, model)
        else:
            resolved_base_url = _resolve_base_url(base_url)
            if _is_chat_completions_style(resolved_base_url):
                llm_result, response_id = _call_openai_chat_completions_api(
                    bundle,
                    model=model,
                    api_key=api_key,
                    base_url=resolved_base_url,
                    timeout_seconds=timeout_seconds,
                )
            else:
                response_body, response_id = _call_openai_responses_api(
                    bundle,
                    model=model,
                    api_key=api_key,
                    base_url=resolved_base_url,
                    timeout_seconds=timeout_seconds,
                )
                llm_result = json.loads(_extract_output_text(response_body))
        return _apply_llm_result(bundle, llm_result, model=model, response_id=response_id)
    except Exception as exc:
        return _failed_bundle(bundle, model=model, error_message=str(exc))


def run_structured_llm_passes(
    bundles: list[RecordReviewBundle],
    *,
    model: str = _DEFAULT_MODEL,
    max_records: int | None = None,
    responder: Callable[[RecordReviewBundle, str], tuple[dict[str, Any], str]] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 60.0,
) -> list[RecordReviewBundle]:
    """Run the first constrained LLM pass over many structured review bundles.

    Args:
        bundles: Deterministic structured review bundles.
        model: Model name to use.
        max_records: Optional cap on how many bundles to send to the model.
        responder: Optional injected responder for tests.
        api_key: Optional explicit API key.
        base_url: Responses API endpoint. When omitted, the value falls back
            to OPENAI_BASE_URL or the default OpenAI endpoint.
        timeout_seconds: HTTP timeout for live calls.

    Returns:
        Updated bundles with completed or failed LLM fields for the selected
        prefix, while leaving later bundles untouched.
    """
    updated: list[RecordReviewBundle] = []
    for index, bundle in enumerate(bundles):
        if max_records is not None and index >= max_records:
            updated.append(bundle)
            continue
        updated.append(run_structured_llm_pass(
            bundle,
            model=model,
            responder=responder,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        ))
    return updated
