"""
Discourse suppression policy - structural cleanup after broad entityhood passes.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster discourse profile] --> B[Possessive guard]
        B --> C{Quote-only unresolved junk shape?}
        C -->|Address-like| D[SuppressReason.QUOTE_ONLY_ADDRESS_LIKE_DISCOURSE]
        C -->|One-token utterance| E[SuppressReason.QUOTE_ONLY_ONE_TOKEN_DISCOURSE]
        C -->|No| F[No discourse suppression]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassificationDecision
from backend.nlp.types import LexiconCategory, MentionCluster, SuppressReason


def unresolved_discourse_suppression_reason(
    cluster: MentionCluster,
    classification: ClassificationDecision,
) -> SuppressReason | None:
    """Return a structural suppression reason for weak unresolved discourse junk.

    This policy runs after classification and broad entityhood scoring have
    already decided that a cluster is plausible enough to survive in principle.
    It exists to catch the narrower class of quote-only unresolved survivors
    that proved noisy in rescue and should be cleaned up before promotion.

    Args:
        cluster: Mention cluster already enriched with discourse evidence.
        classification: Deterministic classification decision for the cluster.

    Returns:
        A discourse-specific SuppressReason when the cluster matches one of the
        proven junk patterns, otherwise ``None``.
    """
    if classification.winning_category != LexiconCategory.UNRESOLVED:
        return None

    # Possessive-backed survivors can still be meaningful unresolved entities,
    # so keep the explicit non-regression guard here even though bare titles
    # are now allowed to fall through to structural suppression.
    if cluster.possessive_support_count > 0:
        return None

    profile = cluster.discourse_profile
    if not profile.quote_only or profile.non_quote_count > 0:
        return None

    if profile.address_like_count > 0:
        return SuppressReason.QUOTE_ONLY_ADDRESS_LIKE_DISCOURSE

    if profile.one_token_utterance_count > 0:
        return SuppressReason.QUOTE_ONLY_ONE_TOKEN_DISCOURSE

    return None


def discourse_suppression_detail(
    cluster: MentionCluster,
    reason: SuppressReason,
) -> str:
    """Render the human-readable suppression detail for a discourse rule.

    Args:
        cluster: Cluster being suppressed.
        reason: Discourse suppression reason chosen upstream.

    Returns:
        Stable explanation string for reports and diagnostics.
    """
    if reason == SuppressReason.QUOTE_ONLY_ADDRESS_LIKE_DISCOURSE:
        return (
            f"'{cluster.normalized_key}' is quote-only unresolved discourse "
            "used like a form of address without non-quote support"
        )
    if reason == SuppressReason.QUOTE_ONLY_ONE_TOKEN_DISCOURSE:
        return (
            f"'{cluster.normalized_key}' is quote-only unresolved discourse "
            "appearing as a one-token utterance without non-quote support"
        )
    raise ValueError(f"Unsupported discourse suppression reason: {reason}")
