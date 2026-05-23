"""
LLM task runner - execute task packets against an OpenAI-compatible endpoint.

.. code-block:: mermaid

    flowchart TD
        A[LLMTaskPacket list] --> B[Iterate packets]
        B --> C{Responder configured?}
        C -->|No| D[Mark skipped]
        C -->|Yes| E[Build prompt and call provider]
        E --> F{Parse JSON from response}
        F -->|OK| G[Normalize and validate]
        F -->|Fail| H[Mark failed]
        G --> I[LLMTaskResult]
        D & H & I --> J[Result list]
"""

from __future__ import annotations

import json
import re
from typing import Callable
from urllib import request
from urllib.error import HTTPError

from backend.nlp.llm_tasks.models import normalize_rescue_payload
from backend.nlp.types import (
    LLMTaskPacket,
    LLMTaskResult,
    LLMTaskResultStatus,
)

Responder = Callable[[LLMTaskPacket, str], tuple[dict, str]]

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _system_prompt() -> str:
    """Return the system prompt for suppression rescue tasks.

    Kept as a standalone function so prompt iteration does not require
    touching the responder or runner code.
    """
    return (
        "You are reviewing a suppressed entity from a fiction manuscript. "
        "The deterministic pipeline filtered this mention, but it may be a real entity. "
        "Examine surrounding manuscript text in the evidence windows. "
        "Your primary task is a binary verdict: is this a genuine named entity "
        "that recurs meaningfully in the narrative? "
        "A genuine entity is a named character, place, ship, group, "
        "title-as-name, or narrative concept. "
        "A common English word, generic descriptor, or structural noise is not. "
        "Do not let classification uncertainty influence your rescue verdict.\n\n"
        "Return exactly one JSON object with no markdown fences, "
        "using this schema:\n"
        "{\n"
        '  "rescue": true or false,\n'
        '  "confidence": number between 0.0 and 1.0,\n'
        '  "rationale": "one sentence explaining the verdict",\n'
        '  "type_hint": "slash-separated from: '
        'character/place/group/object/event/concept/title",\n'
        '  "canonical_name": "best display name or null"\n'
        "}\n\n"
        "type_hint and canonical_name are optional non-authoritative hints. "
        "When the entity fits multiple categories, combine them with slashes "
        "(e.g. \"group/title\"). "
        "Use null for canonical_name if uncertain."
    )


def _build_user_prompt(packet: LLMTaskPacket) -> str:
    """Render the user-facing prompt from one task packet.

    Sends only the fields the model needs: entity metadata, task
    instructions, and evidence windows. Traceability fields like
    source_bundle_kind and source_authority_weight stay out of the
    prompt to keep token usage low for small models.
    """
    prompt_payload = {
        "task_id": packet.task_id,
        "entity": packet.payload,
        "task_goal": packet.task_goal,
        "task_constraints": packet.task_constraints,
        "evidence": [
            {
                "quote": item.quote,
                "context_before": item.context_before,
                "context_after": item.context_after,
                "document_path": item.document_path,
                "suppression_reason": item.suppression_reason,
            }
            for item in packet.evidence_payload
        ],
    }
    return json.dumps(prompt_payload, ensure_ascii=False)


def _extract_json_object(text: str) -> str:
    """Extract a JSON object from a model response body.

    Handles markdown fences and surrounding prose. Raises ValueError
    when no JSON object is found.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = _JSON_OBJECT_PATTERN.search(stripped)
    if match is None:
        raise ValueError("No JSON object found in model response content.")
    return match.group(0)


def make_chat_responder(
    *,
    api_key: str = "",
    base_url: str = "https://integrate.api.nvidia.com/v1",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    timeout_seconds: float = 60.0,
) -> Responder:
    """Build a responder that calls an OpenAI-compatible chat completions endpoint.

    Args:
        api_key: Bearer token for authentication.
        base_url: Base URL for the chat completions API.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum tokens in the response.
        timeout_seconds: HTTP request timeout.

    Returns:
        A responder callable that takes (packet, model) and returns
        (parsed_payload, response_id).
    """
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def responder(packet: LLMTaskPacket, model: str) -> tuple[dict, str]:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _build_user_prompt(packet)},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                pass
            detail = f"HTTP {exc.code}: {exc.reason}"
            if error_body:
                detail = f"{detail} body={error_body[:600]}"
            raise RuntimeError(detail) from exc

        response_id = str(response_body.get("id", ""))
        message = response_body.get("choices", [{}])[0].get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            raise ValueError("Provider response missing choices[0].message.content.")

        extracted = _extract_json_object(content)
        try:
            parsed_payload = json.loads(extracted)
        except json.JSONDecodeError as exc:
            parsed_payload = {
                "raw_model_output": content,
                "raw_extracted_json": extracted,
                "json_parse_error": str(exc),
            }
        return parsed_payload, response_id

    return responder


def run_task_packets(
    packets: list[LLMTaskPacket],
    *,
    model: str,
    provider: str,
    responder: Responder | None = None,
    on_result: Callable[[LLMTaskResult], None] | None = None,
) -> list[LLMTaskResult]:
    """Run task packets and return structured results.

    Args:
        packets: Task packets from rescue assembly.
        model: Model identifier for traceability.
        provider: Provider identifier for traceability.
        responder: Callable that executes one packet against the model.
            When omitted, all packets are marked as skipped.
        on_result: Optional callback after each result for incremental
            persistence during long runs.

    Returns:
        One LLMTaskResult per packet, preserving input order.
    """
    results: list[LLMTaskResult] = []
    for packet in packets:
        if responder is None:
            result = LLMTaskResult(
                task_id=packet.task_id,
                task_family=packet.task_family,
                schema_id=packet.schema_id,
                status=LLMTaskResultStatus.SKIPPED,
                model=model,
                provider=provider,
                error="no responder configured",
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
            continue
        try:
            payload, response_id = responder(packet, model)
            normalized = normalize_rescue_payload(
                payload,
                fallback_normalized_key=packet.source_object_id,
            )
            result = LLMTaskResult(
                task_id=packet.task_id,
                task_family=packet.task_family,
                schema_id=packet.schema_id,
                status=LLMTaskResultStatus.COMPLETED,
                model=model,
                provider=provider,
                response_id=response_id,
                payload=normalized.model_dump(mode="json"),
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
        except Exception as exc:
            result = LLMTaskResult(
                task_id=packet.task_id,
                task_family=packet.task_family,
                schema_id=packet.schema_id,
                status=LLMTaskResultStatus.FAILED,
                model=model,
                provider=provider,
                error=str(exc),
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results
