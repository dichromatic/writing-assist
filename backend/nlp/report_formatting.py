"""Shared formatting helpers for text reports."""

# Diagram omitted - utility module with no significant information flow.

from __future__ import annotations


def hr(title: str = "") -> str:
    """Return one stable report separator line.

    Args:
        title: Optional section title.

    Returns:
        Separator line for report sections.
    """
    width = 72
    if title:
        pad = width - len(title) - 2
        return f"\n-- {title} " + "-" * pad
    return "-" * width
