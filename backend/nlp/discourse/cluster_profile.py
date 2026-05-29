"""
Cluster discourse enrichment - aggregate quote-local usage patterns per cluster.

.. code-block:: mermaid

    flowchart TD
        A[PreprocessedDocument] --> C[Quote and sentence indexes]
        B[MentionCluster anchors] --> D[Per-anchor discourse checks]
        C --> D
        D --> E[ClusterDiscourseProfile]
        E --> F[Mutated MentionCluster list]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.discourse.address_like import is_address_like_reference
from backend.nlp.types import (
    ClusterDiscourseProfile,
    MentionCluster,
    PreprocessedDocument,
)


def _quote_ranges_by_span(
    pre: PreprocessedDocument,
) -> dict[int, list[tuple[int, int]]]:
    """Index quote ranges by span for fast anchor containment checks.

    Args:
        pre: Preprocessed document containing quote spans.

    Returns:
        Mapping from span ordinal to enclosing quote ranges.
    """
    ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for quote in pre.quote_spans:
        ranges[quote.span_ordinal].append((quote.start_char, quote.end_char))
    return dict(ranges)


def _sentences_by_span(pre: PreprocessedDocument) -> dict[int, list]:
    """Index preprocessed sentences by span ordinal.

    Args:
        pre: Preprocessed document containing sentence records.

    Returns:
        Mapping from span ordinal to sentences in that span.
    """
    sentences: dict[int, list] = defaultdict(list)
    for sentence in pre.sentences:
        sentences[sentence.span_ordinal].append(sentence)
    return dict(sentences)


def _count_anchor_one_token_utterance(
    sentence,
    anchor,
    enclosing_quotes: list[tuple[int, int]],
) -> int:
    """Return 1 when an anchor is the only lexical token in its quote.

    This intentionally mirrors the current record-side interpretation so the
    Stage 1 migration changes where the signal is used, not what it means.

    Args:
        sentence: Sentence containing the anchor.
        anchor: Span anchor under inspection.
        enclosing_quotes: Quote ranges that fully contain the anchor.

    Returns:
        ``1`` when the anchor is the lone lexical token inside a quote,
        otherwise ``0``.
    """
    for quote_start, quote_end in enclosing_quotes:
        quote_tokens = [
            token
            for token in sentence.tokens
            if quote_start <= token.start_char
            and token.end_char <= quote_end
            and any(character.isalpha() for character in token.text)
        ]
        if len(quote_tokens) != 1:
            continue
        token = quote_tokens[0]
        if token.start_char == anchor.start_char and token.end_char == anchor.end_char:
            return 1
    return 0


def build_cluster_discourse_profile(
    cluster: MentionCluster,
    *,
    quote_ranges_by_span: dict[int, list[tuple[int, int]]],
    sentences_by_span: dict[int, list],
) -> ClusterDiscourseProfile:
    """Aggregate minimal discourse evidence for one mention cluster.

    Args:
        cluster: Mention cluster whose anchors will be inspected.
        quote_ranges_by_span: Quote ranges indexed by span ordinal.
        sentences_by_span: Sentences indexed by span ordinal.

    Returns:
        Minimal cluster-side discourse profile for Stage 1 entityhood use.
    """
    in_quote_count = 0
    non_quote_count = 0
    address_like_count = 0
    one_token_utterance_count = 0

    for anchor in cluster.anchors:
        quote_ranges = quote_ranges_by_span.get(anchor.span_ordinal, [])
        enclosing_quotes = [
            (start, end)
            for start, end in quote_ranges
            if start <= anchor.start_char and anchor.end_char <= end
        ]
        if enclosing_quotes:
            in_quote_count += 1
        else:
            non_quote_count += 1

        for sentence in sentences_by_span.get(anchor.span_ordinal, []):
            if not (
                sentence.start_char <= anchor.start_char
                and anchor.end_char <= sentence.end_char
            ):
                continue
            try:
                token_index = next(
                    index
                    for index, token in enumerate(sentence.tokens)
                    if token.start_char == anchor.start_char
                )
            except StopIteration:
                break

            if is_address_like_reference(sentence, token_index, quote_ranges):
                address_like_count += 1
            one_token_utterance_count += _count_anchor_one_token_utterance(
                sentence,
                anchor,
                enclosing_quotes,
            )
            break

    return ClusterDiscourseProfile(
        in_quote_count=in_quote_count,
        non_quote_count=non_quote_count,
        address_like_count=address_like_count,
        one_token_utterance_count=one_token_utterance_count,
        quote_only=(in_quote_count > 0 and non_quote_count == 0),
    )


def enrich_clusters_with_discourse(
    pre: PreprocessedDocument,
    clusters: list[MentionCluster],
) -> None:
    """Populate minimal discourse profiles on clusters in place.

    Args:
        pre: Preprocessed document for quote and sentence access.
        clusters: Mention clusters to mutate with discourse evidence.
    """
    quote_ranges_by_span = _quote_ranges_by_span(pre)
    sentences_by_span = _sentences_by_span(pre)
    for cluster in clusters:
        cluster.discourse_profile = build_cluster_discourse_profile(
            cluster,
            quote_ranges_by_span=quote_ranges_by_span,
            sentences_by_span=sentences_by_span,
        )
