# Diagram omitted - utility module with no significant information flow.

"""
Helpers for reasoning about multi-token compound entity keys.

These helpers centralize the simple shape rules used by multiple
classification scorers so compound handling stays consistent across
categories.
"""

from __future__ import annotations

from backend.nlp.types import MentionCluster


def compound_parts(cluster: MentionCluster) -> tuple[str, ...]:
    """Return normalized key parts for a compound-aware cluster.

    Args:
        cluster: Cluster whose normalized key may contain multiple tokens.

    Returns:
        Tuple of normalized key parts in surface order.
    """
    return tuple(part for part in cluster.normalized_key.split() if part)


def compound_head(cluster: MentionCluster) -> str | None:
    """Return the final token of a multi-token normalized key.

    Args:
        cluster: Cluster whose compound head may carry category semantics.

    Returns:
        The final normalized key token for multi-token compounds, otherwise
        ``None`` for single-token clusters.
    """
    parts = compound_parts(cluster)
    if len(parts) < 2:
        return None
    return parts[-1]
