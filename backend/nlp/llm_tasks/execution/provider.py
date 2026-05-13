"""
Shared LLM task runner - execute task packets outside deterministic inspection CLIs.

.. code-block:: mermaid

    flowchart TD
        A[LLMTaskPacket list] --> B[Iterate packets]
        B --> C{Responder available}
        C -->|No| D[Mark skipped]
        C -->|Yes| E[Call provider responder]
        E --> F{Call success}
        F -->|Yes| G[Completed LLMTaskResult]
        F -->|No| H[Failed LLMTaskResult]
        D & G & H --> I[LLMTaskResult list]
"""

from __future__ import annotations

import json
import re
from urllib import request
from urllib.error import HTTPError
from typing import Callable

from backend.nlp.llm_tasks.execution.models import normalize_llm_payload
from backend.nlp.types import (
    LLMTaskPassStage,
    LLMTaskPacket,
    LLMTaskResult,
    LLMTaskResultStatus,
)

Responder = Callable[[LLMTaskPacket, str], tuple[dict, str]]

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _system_prompt_for_packet(packet: LLMTaskPacket) -> str:
    """Return task-family specific system prompt instructions."""
    base = (
        "Return only one JSON object. Do not include markdown fences. "
        "Ground every output claim in provided evidence payload items."
    )
    if packet.task_family.value == "record_fact_extraction":
        return (
            base
            + " Extract atomic lore facts only. "
            + "Do not output section headings, labels, or record metadata as facts. "
            + "Do not copy full paragraphs verbatim when they contain multiple claims; split into atomic statements."
            + " Never emit facts like 'Section heading is ...', 'record type is ...', or restatements of heading/title lines."
            + " If an evidence item is a heading label, use it only to contextualize adjacent prose facts, not as its own fact."
            + " Example good decomposition: "
            + "'History is a convergence of three arcs' + "
            + "'Arc one is common-lineage civilization' + "
            + "'Arc two is Lunarian custodianship' + "
            + "'Arc three is magician lineage'. "
            + " Example bad decomposition: one paragraph-long restatement of the whole source text."
        )
    if packet.task_family.value == "manuscript_entity_profile":
        return (
            base
            + " Build an entity profile from evidence snippets and context windows. "
            + "When evidence is sparse or role-title only, keep uncertainty explicit."
        )
    if packet.task_family.value == "manuscript_reference_attachment":
        return (
            base
            + " Propose attachment targets conservatively and keep unresolved outcomes explicit."
        )
    if packet.task_family.value == "manuscript_category_resolution":
        return (
            base
            + " Resolve category only when evidence supports it; otherwise keep review_required true."
        )
    if packet.task_family.value == "manuscript_entity_review_resolution":
        return (
            base
            + " You are resolving a flagged entity from a previous review pass. "
            + "The prior pass profile, uncertainty reason, and broader scene context are provided. "
            + "Resolve category and identity only when evidence supports it. "
            + "If evidence remains insufficient, keep review_required true and explain what is missing."
        )
    return base


def _extract_json_object_text(text: str) -> str:
    """Extract the first JSON object from a model response body."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()
    match = _JSON_BLOCK_PATTERN.search(stripped)
    if match is None:
        raise ValueError("No JSON object found in model response content.")
    return match.group(0)


def _packet_prompt(packet: LLMTaskPacket) -> str:
    """Render one packet into a structured prompt payload string."""
    payload = {
        "task_id": packet.task_id,
        "task_family": packet.task_family.value,
        "schema_id": packet.schema_id,
        "source_bundle_kind": packet.source_bundle_kind,
        "source_object_kind": packet.source_object_kind,
        "source_object_id": packet.source_object_id,
        "source_document_paths": packet.source_document_paths,
        "document_type": packet.document_type.value,
        "document_status": packet.document_status.value,
        "source_authority": packet.source_authority,
        "source_authority_weight": packet.source_authority_weight,
        "task_goal": packet.task_goal,
        "task_constraints": packet.task_constraints,
        "selection_reason": packet.selection_reason,
        "payload": packet.payload,
        "evidence_payload": [
            {
                "evidence_id": item.evidence_id,
                "document_path": item.document_path,
                "source_anchor": {
                    "path": item.source_anchor.path,
                    "span_ordinal": item.source_anchor.span_ordinal,
                    "start_char": item.source_anchor.start_char,
                    "end_char": item.source_anchor.end_char,
                },
                "quote": item.quote,
                "context_before": item.context_before,
                "context_after": item.context_after,
                "source_object_id": item.source_object_id,
                "visibility_bucket": item.visibility_bucket,
                "suppression_reason": item.suppression_reason,
                "confidence_score": item.confidence_score,
                "evidence_metadata": item.evidence_metadata,
            }
            for item in packet.evidence_payload
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def make_nvidia_nim_chat_responder(
    *,
    api_key: str,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    timeout_seconds: float = 60.0,
    prompt_variant: str = "baseline",
) -> Responder:
    """Build a responder that calls NVIDIA NIM chat completions."""

    normalized_base = base_url.rstrip("/")
    endpoint = f"{normalized_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    def responder(packet: LLMTaskPacket, model: str) -> tuple[dict, str]:
        system_prompt = _system_prompt_for_packet(packet)
        if packet.task_family.value == "manuscript_entity_profile":
            normalized_variant = prompt_variant.strip().lower()
            if normalized_variant == "downgraded":
                system_prompt += (
                    " Treat deterministic category hints as non-authoritative priors. "
                    "Prefer unresolved when context does not clearly support a category. "
                    "If review_required is true, include uncertainty_reason."
                )
            elif normalized_variant == "refute_first":
                system_prompt += (
                    " Use refutation-first verification: try to disconfirm deterministic assumptions first. "
                    "Only keep deterministic category when evidence context explicitly supports it. "
                    "If contradiction or insufficiency exists, set review_required true and include uncertainty_reason."
                )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _packet_prompt(packet)},
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
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            detail = f"HTTP {exc.code}: {exc.reason}"
            if body:
                detail = f"{detail} body={body[:600]}"
            raise RuntimeError(detail) from exc
        response_id = str(response_body.get("id", ""))
        message = response_body.get("choices", [{}])[0].get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            raise ValueError("Provider response missing choices[0].message.content.")
        extracted = _extract_json_object_text(content)
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


def run_llm_task_packets(
    packets: list[LLMTaskPacket],
    *,
    model: str,
    provider: str,
    responder: Responder | None = None,
    pass_stage: LLMTaskPassStage = LLMTaskPassStage.FIRST_PASS,
) -> list[LLMTaskResult]:
    """Run shared LLM task packets and return structured results.

    Args:
        packets: Shared task packets from review-bundle handoff artifacts.
        model: Model identifier requested for execution.
        provider: Provider identifier for reporting and traceability.
        responder: Callable that executes one packet and returns
            `(payload, response_id)`. When omitted, all packets are marked
            as skipped.

    Returns:
        One `LLMTaskResult` per packet, preserving input order.
    """
    results: list[LLMTaskResult] = []
    for packet in packets:
        if responder is None:
            results.append(
                LLMTaskResult(
                    task_id=packet.task_id,
                    task_family=packet.task_family,
                    schema_id=packet.schema_id,
                    status=LLMTaskResultStatus.SKIPPED,
                    model=model,
                    provider=provider,
                    pass_stage=pass_stage,
                    error="no responder configured",
                )
            )
            continue
        try:
            payload, response_id = responder(packet, model)
            normalized = normalize_llm_payload(
                task_family=packet.task_family,
                raw_payload=payload,
            )
            results.append(
                LLMTaskResult(
                    task_id=packet.task_id,
                    task_family=packet.task_family,
                    schema_id=packet.schema_id,
                    status=LLMTaskResultStatus.COMPLETED,
                    model=model,
                    provider=provider,
                    pass_stage=pass_stage,
                    response_id=response_id,
                    payload=normalized.model_dump(mode="json"),
                )
            )
        except Exception as exc:  # pragma: no cover - runtime provider exceptions vary.
            results.append(
                LLMTaskResult(
                    task_id=packet.task_id,
                    task_family=packet.task_family,
                    schema_id=packet.schema_id,
                    status=LLMTaskResultStatus.FAILED,
                    model=model,
                    provider=provider,
                    pass_stage=pass_stage,
                    error=str(exc),
                )
            )
    return results
