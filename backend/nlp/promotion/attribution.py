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

from backend.nlp.types import (
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
    # Map lowercase base surface form -> cluster normalized_key.
    # Possessives are stripped so "Aldous's" resolves to "aldous".
    surface_to_key: dict[str, str] = {}
    for cluster in clusters:
        for surface in cluster.surface_forms:
            base = surface[:-2] if surface.endswith("'s") else surface
            surface_to_key[base.lower()] = cluster.normalized_key

    # Index quotes by span_ordinal to avoid a quadratic scan per sentence.
    quotes_by_span: dict[int, list[QuoteSpan]] = {}
    for quote in pre.quote_spans:
        quotes_by_span.setdefault(quote.span_ordinal, []).append(quote)

    records: list[AttributionRecord] = []

    for sentence in pre.sentences:
        span_quotes = quotes_by_span.get(sentence.span_ordinal, [])

        for quote in span_quotes:
            # Skip quotes that fall outside this sentence's character range.
            # A span can contain multiple sentences, so we must filter by char
            # position rather than span_ordinal alone.
            if not (sentence.start_char <= quote.start_char < sentence.end_char):
                continue

            pre_tokens = [t for t in sentence.tokens if t.end_char <= quote.start_char]
            post_tokens = [t for t in sentence.tokens if t.start_char >= quote.end_char]

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
    """Return the normalized_key of the speaker cluster nearest to the speech verb.

    Requires both a speech verb (in any inflected form) and a known cluster
    surface in the token window. When multiple cluster surfaces are present,
    the one whose closest token is nearest (in token distance) to any speech
    verb in the window wins. This correctly handles sentences where an observer
    appears before the actual speaker, e.g. "As Aldous watched, Mary said, '...'".

    Distance is measured from the nearest token within the surface to the
    nearest speech verb, so a multi-word surface like "Captain Marsh" at
    positions [2, 3] with "said" at position 4 has distance 1, not 2.

    Args:
        tokens: The pre-quote or post-quote token window, already size-limited.
        surface_to_key: Mapping from lowercase base surface to normalized_key.

    Returns:
        The normalized_key of the attributed speaker, or None if no match found.
    """
    token_texts = [t.text.lower() for t in tokens]

    # Collect all speech verb positions so multi-verb windows are handled
    # correctly (e.g. "asked and replied"). Distance is measured to the nearest.
    verb_positions = [i for i, t in enumerate(token_texts) if _is_speech_verb(t)]
    if not verb_positions:
        return None

    best_key: Optional[str] = None
    best_distance: float = float('inf')

    for surface, key in surface_to_key.items():
        words = surface.split()
        for i in range(len(token_texts) - len(words) + 1):
            if token_texts[i:i + len(words)] == words:
                # Use the surface token closest to any verb; this gives
                # multi-word surfaces a fair comparison with single-word ones.
                distance = min(
                    abs(surface_pos - verb_pos)
                    for surface_pos in range(i, i + len(words))
                    for verb_pos in verb_positions
                )
                if distance < best_distance:
                    best_distance = distance
                    best_key = key
                # Only consider the first occurrence of each surface in the
                # window; a repeated name does not change the speaker.
                break

    return best_key

    return None
