"""
Shared LLM task builder dispatcher.

.. code-block:: mermaid

    flowchart TD
        A[Deterministic review artifacts] --> B{Bundle kind}
        B -->|structured records| C[Structured-record task builder]
        B -->|manuscript| D[Manuscript task builder]
        C & D --> E[LLMTaskPacket list]
        C & D --> F[LLMTaskSelectionDiagnostic list]
"""

from __future__ import annotations

from backend.nlp.llm_tasks.assembly.manuscripts import build_manuscript_task_packets
from backend.nlp.llm_tasks.assembly.structured_records import build_structured_record_task_packets
from backend.nlp.types import LLMTaskPacket, LLMTaskSelectionDiagnostic, ManuscriptReviewBundle, RecordReviewBundle


def build_llm_task_packets(
    *,
    record_review_bundles: list[RecordReviewBundle] | None = None,
    manuscript_review_bundle: ManuscriptReviewBundle | None = None,
) -> tuple[list[LLMTaskPacket], list[LLMTaskSelectionDiagnostic]]:
    """Build shared LLM task packets from deterministic review artifacts."""
    packets: list[LLMTaskPacket] = []
    diagnostics: list[LLMTaskSelectionDiagnostic] = []
    if record_review_bundles is not None:
        record_packets, record_diagnostics = build_structured_record_task_packets(record_review_bundles)
        packets.extend(record_packets)
        diagnostics.extend(record_diagnostics)
    if manuscript_review_bundle is not None:
        manuscript_packets, manuscript_diagnostics = build_manuscript_task_packets(manuscript_review_bundle)
        packets.extend(manuscript_packets)
        diagnostics.extend(manuscript_diagnostics)
    return packets, diagnostics
