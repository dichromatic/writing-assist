"""
Shared token-span lookup helpers for classification modules.

Compound entity harvesting means a cluster anchor may span multiple adjacent
tokens rather than mapping 1:1 to a single token. Classifiers need a common
way to recover the token window that begins at the anchor start and extends to
the anchor end. The same anchor iteration is also reused by several scorers
that need to ask "does any anchor satisfy this local token pattern?" without
repeating the scan loop in each module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from backend.nlp.types import MentionCluster, PreprocessedDocument, Token


def iter_anchor_token_starts(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
) -> Iterator[tuple[list[Token], int]]:
    """Yield the token list and starting token index for each cluster anchor.

    For single-token anchors this behaves like the previous exact-token lookup.
    For compound anchors, it returns the index of the first token covered by
    the anchor as long as some later token in the same span ends exactly at the
    anchor end.
    """
    if pre is None:
        return

    for anchor in cluster.anchors:
        tokens = pre.tokens_by_span.get(anchor.span_ordinal, [])
        for index, token in enumerate(tokens):
            if token.start_char != anchor.start_char:
                continue

            for tail_index in range(index, len(tokens)):
                if tokens[tail_index].end_char == anchor.end_char:
                    yield tokens, index
                    break
            break


def has_anchor_token_pattern(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
    check_fn: Callable[[list[Token], int], bool],
) -> bool:
    """Return True when any cluster anchor satisfies a token-context check.

    Args:
        cluster: Cluster whose anchors should be evaluated.
        pre: Preprocessed document context that can recover token spans.
        check_fn: Predicate applied to each `(tokens, index)` anchor start.

    Returns:
        True when the predicate matches at least one anchor start.
    """
    for tokens, index in iter_anchor_token_starts(cluster, pre):
        if check_fn(tokens, index):
            return True
    return False
