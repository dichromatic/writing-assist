"""
Preprocessing - tokenise, sentence-split, and quote-detect each parsed span.

All character normalisations are 1-to-1 substitutions so that every token's
`start_char`/`end_char` indices are valid positions in the *original* text.
This means callers can safely slice `doc.raw_text[token.start_char:token.end_char]`
and recover the exact original surface form without a separate mapping table.

.. code-block:: mermaid

    flowchart TD
        A[ParsedMarkdownDocument] --> B[For each content span]
        B --> C[Normalise span text\n1:1 char substitutions]
        C --> D[Tokenise normalised text\noffsets valid in original]
        D --> E[Split into sentences\nabbreviation-aware]
        D --> F[Detect quote spans\npaired double- and single-quotes]
        B --> G[Tag structural markers\nheadings and scene breaks]
        E & F & G --> H[PreprocessedDocument]
"""

from __future__ import annotations

import re
from typing import Optional

from backend.nlp.types import (
    Heading,
    Paragraph,
    ParsedMarkdownDocument,
    PreprocessedDocument,
    QuoteSpan,
    SceneBreak,
    Sentence,
    SpanAnchor,
    StructuralMarker,
    StructuralMarkerKind,
    Token,
)

# ---------------------------------------------------------------------------
# 1:1 character normalisations
#
# Every substitution maps one Unicode code point to exactly one ASCII code
# point. The normalised string therefore has the same length (in Python
# characters) as the original, so token offsets computed against the
# normalised string are identical to offsets into the original string.
# ---------------------------------------------------------------------------
_CHAR_NORMALIZATIONS = str.maketrans({
    '“': '"',   # LEFT DOUBLE QUOTATION MARK  ->  "
    '”': '"',   # RIGHT DOUBLE QUOTATION MARK ->  "
    '‘': "'",   # LEFT SINGLE QUOTATION MARK  ->  '
    '’': "'",   # RIGHT SINGLE QUOTATION MARK ->  '
    'ʼ': "'",   # MODIFIER LETTER APOSTROPHE  ->  '
    '—': '-',   # EM DASH                     ->  -
    '–': '-',   # EN DASH                     ->  -
    '…': '.',   # HORIZONTAL ELLIPSIS         ->  .
    ' ': ' ',   # NO-BREAK SPACE              ->  space
})

# Matches a word (possibly with a contraction apostrophe) or a single
# non-whitespace character. The contraction sub-pattern `(?:'\w+)*` ensures
# that "don't" is one token rather than ["don", "'", "t"].
_TOKEN_RE = re.compile(r"\w+(?:'\w+)*|\S")

# Sentence-terminal punctuation that can end a sentence.
_SENTENCE_TERMINALS = frozenset('.!?')

# Title abbreviations that end with a period but do NOT end a sentence.
# Expanding this list is the only change needed to handle new abbreviations.
_ABBREV_TITLES = frozenset({
    'Dr', 'Mr', 'Mrs', 'Ms', 'Miss', 'Prof', 'Rev',
    'Capt', 'Cpt', 'Lt', 'Sgt', 'Col', 'Gen', 'Adm', 'St',
})


def _normalize_span_text(text: str) -> str:
    """Apply 1:1 character substitutions to produce a normalised copy.

    The returned string has the same length as `text` (in Python characters),
    so any index into the returned string is also a valid index into `text`.

    Args:
        text: Raw span text, as stored on Heading or Paragraph.

    Returns:
        Normalised text with typographic punctuation replaced by ASCII equivalents.
    """
    return text.translate(_CHAR_NORMALIZATIONS)


def _tokenize(
    original: str,
    normalized: str,
    span_start: int,
    span_ordinal: int,
) -> list[Token]:
    """Tokenise `normalized` and build Token records whose offsets point into `original`.

    Because `normalized` was produced by 1:1 substitutions, every index in
    `normalized` is the same character position as in `original`. The
    `raw_text` field of each token is sliced from `original`, preserving the
    source characters exactly.

    Args:
        original: The unmodified span text (used to recover raw token surfaces).
        normalized: The normalised span text (used for regex matching).
        span_start: Absolute character offset of this span's first character
            within the document's raw_text.
        span_ordinal: The span ordinal of the parent span.

    Returns:
        Tokens in document order with absolute character offsets.
    """
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(normalized):
        rel_start = m.start()
        rel_end = m.end()
        abs_start = span_start + rel_start
        abs_end = span_start + rel_end
        tokens.append(Token(
            text=m.group(),
            raw_text=original[rel_start:rel_end],
            start_char=abs_start,
            end_char=abs_end,
            span_ordinal=span_ordinal,
        ))
    return tokens


def _split_sentences(tokens: list[Token], span_ordinal: int) -> list[Sentence]:
    """Group tokens into sentences using terminal-punctuation heuristics.

    A new sentence begins after a `.`, `!`, or `?` token when the *next*
    non-punctuation token starts with an uppercase letter - unless the terminal
    is a `.` that follows a known title abbreviation, which is an abbreviation
    period rather than a sentence boundary.

    Args:
        tokens: All tokens for a single span, in document order.
        span_ordinal: The span ordinal used for all sentence anchor fields.

    Returns:
        Sentences in document order. Always at least one sentence when tokens
        is non-empty. Returns an empty list when tokens is empty.
    """
    if not tokens:
        return []

    sentences: list[Sentence] = []
    sentence_start = 0

    def close_sentence(end_idx: int) -> None:
        span_tokens = tokens[sentence_start:end_idx]
        if not span_tokens:
            return
        sentences.append(Sentence(
            tokens=span_tokens,
            start_char=span_tokens[0].start_char,
            end_char=span_tokens[-1].end_char,
            span_ordinal=span_ordinal,
        ))

    for i, token in enumerate(tokens):
        if token.text not in _SENTENCE_TERMINALS:
            continue

        # Find the next token that is not itself punctuation to check case.
        next_word_idx = i + 1
        while next_word_idx < len(tokens) and not tokens[next_word_idx].text[0].isalpha():
            next_word_idx += 1

        if next_word_idx >= len(tokens):
            # No following word - this terminal ends only the final sentence,
            # which is handled by close_sentence(len(tokens)) below.
            continue

        next_token = tokens[next_word_idx]
        if not next_token.text[0].isupper():
            # Lowercase continuation - not a sentence boundary.
            continue

        if token.text == '.':
            # A period after a title abbreviation is not a sentence boundary.
            prev_word = None
            for j in range(i - 1, -1, -1):
                if tokens[j].text[0].isalpha():
                    prev_word = tokens[j].text
                    break
            if prev_word and prev_word in _ABBREV_TITLES:
                continue

        close_sentence(i + 1)
        sentence_start = i + 1

    close_sentence(len(tokens))
    return sentences


def _detect_quotes(
    path: str,
    tokens: list[Token],
    span_ordinal: int,
    raw_text: str,
) -> list[QuoteSpan]:
    """Pair quote tokens into QuoteSpan records for both double and single quotes.

    Handles `"` (universal double-quote dialogue) and `'` (British-style dialogue
    and internal character speech). Each style is matched independently: a `"`
    opening can only close with `"`, and a `'` opening can only close with `'`.
    If an odd number of tokens of either style appear, the final unmatched token
    is silently discarded - unbalanced quotes are common in prose and should not
    halt preprocessing.

    Apostrophes within contractions ("don't") and possessives ("Aldous's") are
    absorbed by the tokenizer into a single word token, so they never appear as
    standalone `'` tokens. Plural possessives ending with a bare apostrophe
    ("James'") do produce a standalone `'` token, because there is no following
    word character for the contraction sub-pattern to absorb. These are filtered
    in `_pair_quote_tokens` by checking the character immediately before the
    token in the raw text: a possessive apostrophe is always attached to its
    word, so the preceding character is alphanumeric.

    Args:
        path: Document path, used to construct QuoteSpan anchors.
        tokens: All tokens for a single span.
        span_ordinal: The span ordinal of the parent span.
        raw_text: The full document raw text, used to extract inner_text.

    Returns:
        QuoteSpan records for each matched open/close pair, sorted by start_char.
    """
    all_spans: list[QuoteSpan] = []
    for quote_char in ('"', "'"):
        all_spans.extend(
            _pair_quote_tokens(path, tokens, span_ordinal, raw_text, quote_char)
        )
    all_spans.sort(key=lambda q: q.start_char)
    return all_spans


def _pair_quote_tokens(
    path: str,
    tokens: list[Token],
    span_ordinal: int,
    raw_text: str,
    quote_char: str,
) -> list[QuoteSpan]:
    """Pair all standalone tokens matching quote_char into QuoteSpan records.

    For single quotes, tokens immediately following a word character are
    skipped: they are possessive suffixes ("James'"), not quote delimiters.
    The check uses the raw document character at start_char - 1, because the
    tokenizer places the bare apostrophe as a separate token only when there
    is no following word character to absorb into a contraction.

    Args:
        path: Document path for anchor construction.
        tokens: All tokens for a single span.
        span_ordinal: The span ordinal of the parent span.
        raw_text: The full document raw text for inner_text extraction.
        quote_char: The exact character to pair, either `"` or `'`.

    Returns:
        QuoteSpan records for each matched pair, in document order.
    """
    quote_spans: list[QuoteSpan] = []
    open_token: Optional[Token] = None

    for token in tokens:
        if token.text != quote_char:
            continue
        # A bare ' immediately following an alphanumeric character is a
        # possessive suffix, not a quote delimiter. Double quotes have no
        # equivalent ambiguity so this check is single-quote only.
        if (
            quote_char == "'"
            and token.start_char > 0
            and raw_text[token.start_char - 1].isalnum()
        ):
            continue
        if open_token is None:
            open_token = token
        else:
            inner_text = raw_text[open_token.end_char:token.start_char]
            quote_spans.append(QuoteSpan(
                inner_text=inner_text,
                start_char=open_token.start_char,
                end_char=token.end_char,
                span_ordinal=span_ordinal,
                anchor=SpanAnchor(
                    path=path,
                    span_ordinal=span_ordinal,
                    start_char=open_token.start_char,
                    end_char=token.end_char,
                ),
            ))
            open_token = None

    return quote_spans


def _make_structural_marker(
    span: Heading | SceneBreak,
    raw_text: str,
) -> StructuralMarker:
    """Build a StructuralMarker for a span that carries structural meaning.

    Headings and scene breaks are the two span types that produce structural
    markers. Paragraphs carry no structural marker.

    Args:
        span: A Heading or SceneBreak span.
        raw_text: The full document raw text, used to extract the actual scene
            break characters (---, ***, or ___) for SceneBreak spans.

    Returns:
        A StructuralMarker for the span.
    """
    if isinstance(span, Heading):
        return StructuralMarker(
            kind=StructuralMarkerKind.HEADING,
            text=span.text,
            start_char=span.start_char,
            end_char=span.end_char,
            span_ordinal=span.span_ordinal,
        )
    # SceneBreak has no text field; slice the actual marker characters from the
    # raw document so the StructuralMarker faithfully reflects ---, ***, or ___.
    marker_text = raw_text[span.start_char:span.end_char].strip()
    return StructuralMarker(
        kind=StructuralMarkerKind.SCENE_BREAK,
        text=marker_text,
        start_char=span.start_char,
        end_char=span.end_char,
        span_ordinal=span.span_ordinal,
    )


def preprocess(doc: ParsedMarkdownDocument) -> PreprocessedDocument:
    """Tokenise, sentence-split, and quote-detect every content span in `doc`.

    All character offsets on the returned tokens are absolute positions into
    `doc.raw_text`, so callers may slice the raw text directly using token
    anchors without re-parsing.

    Args:
        doc: A fully parsed Markdown document from the parsing stage.

    Returns:
        PreprocessedDocument with tokens, sentences, quote spans, and structural
        markers derived from all content spans.
    """
    tokens_by_span: dict[int, list[Token]] = {}
    all_sentences: list[Sentence] = []
    all_quotes: list[QuoteSpan] = []
    structural_markers: list[StructuralMarker] = []

    content_spans = sorted(
        [*doc.headings, *doc.paragraphs, *doc.scene_breaks],
        key=lambda s: s.span_ordinal,
    )

    for span in content_spans:
        if isinstance(span, (Heading, SceneBreak)):
            structural_markers.append(_make_structural_marker(span, doc.raw_text))

        # SceneBreak spans carry no text to tokenise.
        if isinstance(span, SceneBreak):
            continue

        original = span.text
        normalized = _normalize_span_text(original)
        tokens = _tokenize(original, normalized, span.start_char, span.span_ordinal)
        sentences = _split_sentences(tokens, span.span_ordinal)
        quotes = _detect_quotes(doc.path, tokens, span.span_ordinal, doc.raw_text)

        tokens_by_span[span.span_ordinal] = tokens
        all_sentences.extend(sentences)
        all_quotes.extend(quotes)

    return PreprocessedDocument(
        source=doc,
        sentences=all_sentences,
        quote_spans=all_quotes,
        structural_markers=structural_markers,
        tokens_by_span=tokens_by_span,
    )
