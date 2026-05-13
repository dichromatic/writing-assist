"""
Review queue report renderer.

.. code-block:: mermaid

    flowchart TD
        A[ReviewQueueItem list] --> B[Summarize queue counts]
        A --> C[Render per-item priors and uncertainty]
        A --> D[Render top snippets]
        B & C & D --> E[Human-readable review queue report]
"""

from __future__ import annotations

from backend.nlp.text_filtering import strip_emoji
from backend.nlp.llm_tasks.review.review_queue import ReviewQueueItem


def _hr(title: str = "") -> str:
    """Return one stable report separator line."""
    width = 72
    if title:
        pad = width - len(title) - 2
        return f"\n-- {title} " + "-" * pad
    return "-" * width


def _truncate(text: str, limit: int = 220) -> str:
    """Return one-line truncated text for compact display."""
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def render_review_queue_report(
    items: list[ReviewQueueItem],
    *,
    max_items: int = 24,
    max_snippets_per_item: int = 3,
) -> str:
    """Render a human-readable report for review queue inspection."""
    lines: list[str] = []
    lines.append(_hr("REVIEW QUEUE"))
    lines.append(f"  item_count: {len(items)}")
    if not items:
        lines.append("  None.")

    for item in items[:max_items]:
        lines.append(_hr(item.canonical_key))
        lines.append(f"  queue_id: {item.queue_id}")
        lines.append(f"  task_id: {item.task_id}")
        lines.append(f"  source_documents: {len(item.source_document_paths)}")
        lines.append(f"  proposed_category: {item.first_pass_assessment.get('proposed_category', '')}")
        lines.append(f"  review_required: {item.first_pass_assessment.get('review_required', False)}")
        lines.append(
            "  uncertainty_reason: "
            + _truncate(str(item.first_pass_assessment.get("uncertainty_reason", "")))
        )
        lines.append(
            "  deterministic_prior: "
            + _truncate(str(item.deterministic_prior))
        )
        lines.append(f"  evidence_snippets: {len(item.evidence_snippets)}")
        for snippet in item.evidence_snippets[:max_snippets_per_item]:
            lines.append(
                "    - "
                f"{snippet.get('evidence_id','')} | {snippet.get('document_path','')} | "
                f"{_truncate(str(snippet.get('quote','')), 120)}"
            )
            lines.append(
                "      ctx: "
                + _truncate(
                    f"{snippet.get('context_before','')} <MENTION> {snippet.get('context_after','')}",
                    220,
                )
            )
    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")
