"""
Dialogue attribution - detects speaker-to-quote relationships in prose.

For each quote span in the preprocessed document, the surrounding sentence is
examined for speech verbs and cluster surface forms. A match produces an
AttributionRecord linking the speaker cluster to the quote.

Attribution is a known weak point in the pipeline. Fiction writing styles are
wildly diverse and will not always follow the patterns this module detects
(post-quote "Name said" or pre-quote "Name said,"). Coverage and false-positive
rate should be validated against real manuscript samples during integration
testing before results are relied on.

.. code-block:: mermaid

    flowchart TD
        A[PreprocessedDocument + MentionCluster list] --> B[Index quote spans by span_ordinal]
        A --> C[Build lowercase surface-to-cluster-key lookup]
        B & C --> D[For each sentence]
        D --> E{Sentence contains a quote span?}
        E -->|No| F[Skip]
        E -->|Yes| G[Extract pre-quote and post-quote token windows]
        G --> H[Post-quote: speech verb lemma + speaker surface?]
        G --> I[Pre-quote: speech verb lemma + speaker surface?]
        H & I --> J{Match found?}
        J -->|Yes| K[Emit AttributionRecord]
        J -->|No| F
        K --> L[list of AttributionRecord]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.nlp.classification.arbitration import classify_clusters
from backend.nlp.types import (
    LexiconCategory,
    MentionCluster,
    PreprocessedDocument,
    QuoteSpan,
    SpanAnchor,
    Token,
)

# ---------------------------------------------------------------------------
# Speech verb lemmas
#
# This set contains base-form (lemma) spellings only. All inflected forms
# (said/says/saying, whispered/whispers/whispering, etc.) are reduced to their
# lemma by _is_speech_verb before lookup, so each verb needs only one entry.
# ---------------------------------------------------------------------------

SPEECH_VERB_LEMMAS: frozenset[str] = frozenset({
    'say', 'ask', 'reply', 'answer', 'call', 'shout',
    'whisper', 'cry', 'murmur', 'declare', 'announce',
    'tell', 'explain', 'continue', 'add', 'admit', 'agree',
    'warn', 'suggest', 'insist', 'demand', 'plead',
    'argue', 'laugh', 'sigh', 'mutter', 'think',
    'respond', 'exclaim', 'state', 'note',
})

# Common inflected forms of the speech verbs above, used only when the
# WordNetLemmatizer is unavailable (no network, corpus not downloaded).
# These are the forms most commonly found in prose - past tense and
# third-person singular. Without this fallback, "said" and "says" would
# not be recognised in environments that lack the NLTK wordnet corpus.
_SPEECH_VERB_FALLBACK: frozenset[str] = SPEECH_VERB_LEMMAS | frozenset({
    'said', 'says', 'saying',
    'asked', 'asks', 'asking',
    'replied', 'replies', 'replying',
    'answered', 'answers', 'answering',
    'called', 'calls', 'calling',
    'shouted', 'shouts', 'shouting',
    'whispered', 'whispers', 'whispering',
    'cried', 'cries', 'crying',
    'murmured', 'murmurs', 'murmuring',
    'declared', 'declares', 'declaring',
    'announced', 'announces', 'announcing',
    'told', 'tells', 'telling',
    'explained', 'explains', 'explaining',
    'continued', 'continues', 'continuing',
    'added', 'adds', 'adding',
    'admitted', 'admits', 'admitting',
    'agreed', 'agrees', 'agreeing',
    'warned', 'warns', 'warning',
    'suggested', 'suggests', 'suggesting',
    'insisted', 'insists', 'insisting',
    'demanded', 'demands', 'demanding',
    'pleaded', 'pleads', 'pleading',
    'argued', 'argues', 'arguing',
    'laughed', 'laughs', 'laughing',
    'sighed', 'sighs', 'sighing',
    'muttered', 'mutters', 'muttering',
    'thought', 'thinks', 'thinking',
    'responded', 'responds', 'responding',
    'exclaimed', 'exclaims', 'exclaiming',
    'stated', 'states', 'stating',
    'noted', 'notes', 'noting',
})

# Maximum number of tokens to examine on either side of a quote boundary.
_WINDOW_SIZE = 10

def _init_lemmatizer():
    """Return an NLTK WordNetLemmatizer, downloading wordnet if needed.

    Returns None if NLTK is unavailable, which causes _is_speech_verb to fall
    back to direct lemma lookup (exact match on base forms only). This degrades
    gracefully: past-tense forms in SPEECH_VERB_LEMMAS like 'say' still match
    'say', but inflected forms like 'says' or 'whispered' are missed.

    Returns:
        A WordNetLemmatizer instance, or None if unavailable.
    """
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        # Trigger the corpus load now so any LookupError surfaces here,
        # not in the middle of a pipeline run.
        lem.lemmatize('said', pos='v')
        return lem
    except LookupError:
        pass

    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        # Probe after download attempt - if the download failed (no network),
        # this raises LookupError which is caught and returns None.
        lem.lemmatize('said', pos='v')
        return lem
    except Exception:
        return None


_LEMMATIZER = _init_lemmatizer()


def _is_speech_verb(word: str) -> bool:
    """Return True if word is a speech verb in any inflected form.

    When the lemmatizer is available, reduces word to its base form with
    pos='v' so said/says/saying all map to 'say', then checks SPEECH_VERB_LEMMAS.
    When the lemmatizer is unavailable (no wordnet corpus), falls back to
    _SPEECH_VERB_FALLBACK which includes the most common inflected forms directly.

    Args:
        word: A lowercase token text to check.

    Returns:
        True if the word is a recognised speech verb.
    """
    if _LEMMATIZER is not None:
        return _LEMMATIZER.lemmatize(word, pos='v') in SPEECH_VERB_LEMMAS
    return word in _SPEECH_VERB_FALLBACK


@dataclass(frozen=True)
class AttributionRecord:
    """A detected link between a speaker cluster and a quote span.

    Args:
        speaker_key: normalized_key of the MentionCluster attributed as speaker.
        quote_anchor: SpanAnchor of the QuoteSpan being attributed.
        pattern: 'post_quote' when the speaker follows the closing quote mark,
            'pre_quote' when the speaker precedes the opening mark.
    """

    speaker_key: str
    quote_anchor: SpanAnchor
    pattern: str


def attribute_dialogue(
    pre: PreprocessedDocument,
    clusters: list[MentionCluster],
) -> list[AttributionRecord]:
    """Detect speaker attributions for all quote spans in the document.

    For each quote span, the tokens immediately before and after the quote
    boundary (within the same sentence) are examined for a speech verb and a
    cluster surface form. When both are found in the same token window, an
    AttributionRecord is emitted. Post-quote detection takes priority over
    pre-quote when both would match.

    Args:
        pre: The preprocessed document with quote_spans and sentences populated.
        clusters: Final MentionCluster records from the clustering stage.

    Returns:
        AttributionRecord entries in sentence order, one per attributed quote.
    """
    # Restrict speaker candidates to clusters that are not already classified
    # as strongly non-person-like. Bare names often remain unresolved before
    # attribution itself has a chance to strengthen them, so unresolved keys
    # stay eligible here. Clear groups and other non-speaker categories are
    # excluded.
    classifications = classify_clusters(clusters, pre, [])
    eligible_keys = {
        key
        for key, decision in classifications.items()
        if decision.winning_category in {
            LexiconCategory.CHARACTER,
            LexiconCategory.PLACE,
            LexiconCategory.UNRESOLVED,
        }
    }

    # Map lowercase base surface form -> cluster normalized_key.
    # Possessives are stripped so "Aldous's" resolves to "aldous".
    surface_to_key: dict[str, str] = {}
    for cluster in clusters:
        if cluster.normalized_key not in eligible_keys:
            continue
        for surface in cluster.surface_forms:
            base = surface[:-2] if surface.endswith("'s") else surface
            surface_to_key[base.lower()] = cluster.normalized_key

    # Index quotes by span_ordinal to avoid a quadratic scan per sentence.
    quotes_by_span: dict[int, list[QuoteSpan]] = {}
    for quote in pre.quote_spans:
        quotes_by_span.setdefault(quote.span_ordinal, []).append(quote)

    records: list[AttributionRecord] = []

    for sentence in pre.sentences:
        span_quotes = sorted(
            quotes_by_span.get(sentence.span_ordinal, []),
            key=lambda quote: quote.start_char,
        )

        for index, quote in enumerate(span_quotes):
            # Skip quotes that fall outside this sentence's character range.
            # A span can contain multiple sentences, so we must filter by char
            # position rather than span_ordinal alone.
            if not (sentence.start_char <= quote.start_char < sentence.end_char):
                continue

            # Quote-local windows must stop at neighboring quotes in the same
            # sentence. Without this, the opening word of a later quote can be
            # mistaken for the speaker of an earlier quote, and the first word
            # of an interrupted quote can be mistaken for the speaker of the
            # resumed fragment.
            previous_quote_end = sentence.start_char
            if index > 0:
                previous_quote_end = max(previous_quote_end, span_quotes[index - 1].end_char)

            next_quote_start = sentence.end_char
            if index + 1 < len(span_quotes):
                next_quote_start = min(next_quote_start, span_quotes[index + 1].start_char)

            pre_tokens = [
                token
                for token in sentence.tokens
                if previous_quote_end <= token.start_char and token.end_char <= quote.start_char
            ]
            post_tokens = [
                token
                for token in sentence.tokens
                if quote.end_char <= token.start_char and token.end_char <= next_quote_start
            ]

            # Post-quote pattern: "..." Speaker said.
            # Checked first because it is the more common English pattern and a
            # cleaner signal - the speaker tag immediately follows the quote mark.
            speaker = _find_speaker(post_tokens[:_WINDOW_SIZE], surface_to_key)
            if speaker is not None:
                records.append(AttributionRecord(
                    speaker_key=speaker,
                    quote_anchor=quote.anchor,
                    pattern='post_quote',
                ))
                continue

            # Pre-quote pattern: Speaker said, "..."
            speaker = _find_speaker(pre_tokens[-_WINDOW_SIZE:], surface_to_key)
            if speaker is not None:
                records.append(AttributionRecord(
                    speaker_key=speaker,
                    quote_anchor=quote.anchor,
                    pattern='pre_quote',
                ))

    return records


def _find_speaker(
    tokens: list[Token],
    surface_to_key: dict[str, str],
) -> Optional[str]:
    """Return the normalized_key of the speaker cluster in a local speech tag.

    Requires both a speech verb (in any inflected form) and a known cluster
    surface in the token window. A surface only qualifies when it forms a
    local speech-tag shape with the verb: either "Speaker said" or
    "said Speaker". This keeps attribution tied to actual speaker tags rather
    than to any nearby capitalized fragment inside adverbial tails, quote
    continuations, or descriptive noun phrases.

    Args:
        tokens: The pre-quote or post-quote token window, already size-limited.
        surface_to_key: Mapping from lowercase base surface to normalized_key.

    Returns:
        The normalized_key of the attributed speaker, or None if no match found.
    """
    token_texts = [t.text.lower() for t in tokens]

    # Collect all speech verb positions so multi-verb windows are handled
    # correctly (e.g. "asked and replied").
    verb_positions = [i for i, t in enumerate(token_texts) if _is_speech_verb(t)]
    if not verb_positions:
        return None

    for surface, key in surface_to_key.items():
        words = surface.split()
        for i in range(len(token_texts) - len(words) + 1):
            if token_texts[i:i + len(words)] == words:
                surface_start = i
                surface_end = i + len(words) - 1

                if any(surface_end == verb_pos - 1 for verb_pos in verb_positions):
                    return key
                if any(surface_start == verb_pos + 1 for verb_pos in verb_positions):
                    return key

                # Only consider the first occurrence of each surface in the
                # window; a repeated non-adjacent name does not become a
                # speaker unless it participates directly in a speech tag.
                break

    return None
