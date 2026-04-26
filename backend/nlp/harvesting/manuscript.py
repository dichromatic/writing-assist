"""
Manuscript harvester - extracts MentionCandidate records from prose spans.

Three extraction passes run per span, in priority order. Tokens consumed by a
higher-priority pass are excluded from lower-priority passes so the same name
does not produce duplicate candidates at different granularities.

After all spans are processed, sentence-initial singleton suppression runs as
a global pass: a candidate whose normalised key appears exactly once in the
full document and whose token was the first word of a sentence is discarded,
because a sentence-initial capital provides no evidence beyond the capitalisation
convention itself.

.. code-block:: mermaid

    flowchart TD
        A[PreprocessedDocument] --> B[Build sentence-initial\nposition index]
        B --> C[For each content span\nin ordinal order]
        C --> D[Pass 1: title-prefixed phrases\nTitle + Name words]
        D --> E[Pass 2: possessive tokens\ntoken ending apostrophe-s]
        E --> E2[Pass 2b: two-token possessives\nbase ending s + bare apostrophe]
        E2 --> F[Pass 3: bare capitalised names\nnot stopword, not consumed]
        F --> G[All raw MentionCandidates]
        G --> H[Global suppression\nsentence-initial singletons]
        H --> I[Filtered MentionCandidates]
"""

from __future__ import annotations

from backend.nlp.types import (
    MentionCandidate,
    PreprocessedDocument,
    SpanAnchor,
)
from backend.nlp.harvesting.shared import (
    TITLE_PREFIXES,
    is_stopword,
    normalize_surface,
    stable_hash_id,
)
from backend.nlp.types import Token


def _is_name_word(token: Token) -> bool:
    """Return True if the token looks like the start of a proper name.

    Accepts tokens whose first character is an uppercase letter. This includes
    contractions and possessives (O'Brien, Aldous's) since those patterns
    legitimately appear in proper names.

    Args:
        token: Any token from a preprocessed span.

    Returns:
        True if the token begins with an uppercase alphabetic character.
    """
    return bool(token.text) and token.text[0].isalpha() and token.text[0].isupper()


def _extract_from_span(
    path: str,
    tokens: list[Token],
    span_ordinal: int,
) -> list[MentionCandidate]:
    """Extract raw MentionCandidate records from a single span's tokens.

    Runs three passes in order: title phrases, possessives, bare capitalized
    names. Each pass marks its tokens consumed so lower-priority passes do not
    re-emit the same text.

    Args:
        path: Document path for anchor construction.
        tokens: Tokens for the span, in document order.
        span_ordinal: Ordinal of the parent span.

    Returns:
        MentionCandidate records for this span. May be empty.
    """
    candidates: list[MentionCandidate] = []
    consumed: set[int] = set()

    def make_candidate(
        surface: str,
        start_char: int,
        end_char: int,
        has_title_prefix: bool,
        has_possessive: bool,
        rule_source: str,
    ) -> MentionCandidate:
        anchor = SpanAnchor(
            path=path,
            span_ordinal=span_ordinal,
            start_char=start_char,
            end_char=end_char,
        )
        return MentionCandidate(
            surface=surface,
            normalized=normalize_surface(surface),
            anchor=anchor,
            has_title_prefix=has_title_prefix,
            has_possessive=has_possessive,
            rule_source=rule_source,
            candidate_id=stable_hash_id(path, str(span_ordinal), surface),
        )

    # ------------------------------------------------------------------
    # Pass 1: title-prefixed phrases (Title + one or more name words)
    # "Captain Aldous", "Dr. Smith", "Lord Reinholt Vayne"
    # ------------------------------------------------------------------
    i = 0
    while i < len(tokens):
        if tokens[i].text in TITLE_PREFIXES and i not in consumed:
            j = i + 1
            # Skip a period that follows an abbreviated title (Dr., Mr., etc.)
            if j < len(tokens) and tokens[j].text == '.':
                j += 1
            name_start = j
            while j < len(tokens) and _is_name_word(tokens[j]):
                j += 1
            if j > name_start:
                # At least one name word follows the title: emit the phrase.
                phrase_tokens = [tokens[i]] + tokens[name_start:j]
                surface = ' '.join(t.raw_text for t in phrase_tokens)
                has_possessive = surface.endswith("'s") or surface.endswith("s'")
                candidates.append(make_candidate(
                    surface=surface,
                    start_char=phrase_tokens[0].start_char,
                    end_char=phrase_tokens[-1].end_char,
                    has_title_prefix=True,
                    has_possessive=has_possessive,
                    rule_source='title_prefix',
                ))
                # Mark all tokens in the phrase as consumed.
                for k in range(i, j):
                    consumed.add(k)
        i += 1

    # ------------------------------------------------------------------
    # Pass 2: possessive tokens (token ending in 's, uppercase base)
    # "Aldous's", "O'Brien's" - the tokenizer keeps these as one token.
    # ------------------------------------------------------------------
    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        if not token.text.endswith("'s"):
            continue
        base = token.text[:-2]
        if not base or not base[0].isalpha() or not base[0].isupper():
            continue
        if is_stopword(base):
            continue
        candidates.append(make_candidate(
            surface=token.raw_text,
            start_char=token.start_char,
            end_char=token.end_char,
            has_title_prefix=False,
            has_possessive=True,
            rule_source='possessive',
        ))
        consumed.add(i)

    # ------------------------------------------------------------------
    # Pass 2b: two-token possessive (base ending in 's + bare apostrophe)
    # "James'" tokenises as ["James", "'"] because the tokenizer regex
    # requires at least one \w after the apostrophe. When the base word
    # ends in 's', a bare apostrophe immediately adjacent signals the
    # standard English terminal-s possessive ("soldiers'", "James'").
    # ------------------------------------------------------------------
    for i, token in enumerate(tokens):
        if i in consumed or i + 1 >= len(tokens):
            continue
        next_tok = tokens[i + 1]
        # The apostrophe must be directly attached - no whitespace between.
        if next_tok.text != "'" or next_tok.start_char != token.end_char:
            continue
        if not token.text[0].isalpha() or not token.text[0].isupper():
            continue
        # Only the terminal-s possessive form is handled here; the base must
        # end in 's' so the combined surface ends in "s'".
        if not token.text[-1].lower() == 's':
            continue
        if is_stopword(token.text):
            continue
        candidates.append(make_candidate(
            surface=token.raw_text + next_tok.raw_text,
            start_char=token.start_char,
            end_char=next_tok.end_char,
            has_title_prefix=False,
            has_possessive=True,
            rule_source='possessive',
        ))
        consumed.add(i)
        consumed.add(i + 1)

    # ------------------------------------------------------------------
    # Pass 3: bare capitalised names (not stopword, not yet consumed)
    # "Aldous", "Rhea", "Vayne"
    # ------------------------------------------------------------------
    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        if not token.text[0].isalpha() or not token.text[0].isupper():
            continue
        if is_stopword(token.text):
            continue
        # Require the token to be purely alphabetic (no digits, punctuation).
        # This filters out things like "I'll" (already consumed as contraction)
        # and standalone uppercase letters used as labels.
        if not token.text.replace("'", '').isalpha():
            continue
        candidates.append(make_candidate(
            surface=token.raw_text,
            start_char=token.start_char,
            end_char=token.end_char,
            has_title_prefix=False,
            has_possessive=False,
            rule_source='bare_capitalized',
        ))

    return candidates


def _suppress_sentence_initial_singletons(
    candidates: list[MentionCandidate],
    sentence_initial_chars: set[int],
) -> list[MentionCandidate]:
    """Remove sentence-initial capitalized names that appear only once.

    A sentence-initial capital provides no evidence of a proper name by itself;
    any common word would be capitalised in that position. If a bare_capitalized
    candidate's normalised key appears only once across all candidates AND its
    first character is at a sentence-initial position, it is suppressed.

    Title-prefix and possessive candidates are never suppressed here: the
    pattern itself (title or possessive) is the evidence, not the position.

    Args:
        candidates: All raw candidates across all spans.
        sentence_initial_chars: Absolute char offsets of the first token of
            each sentence in the document.

    Returns:
        Candidates with sentence-initial singletons removed.
    """
    # Count how many times each normalised key appears across all candidates.
    key_counts: dict[str, int] = {}
    for c in candidates:
        key_counts[c.normalized] = key_counts.get(c.normalized, 0) + 1

    result: list[MentionCandidate] = []
    for c in candidates:
        if (
            c.rule_source == 'bare_capitalized'
            and key_counts[c.normalized] == 1
            and c.anchor.start_char in sentence_initial_chars
        ):
            # Singleton at sentence-initial position with no other evidence.
            continue
        result.append(c)
    return result


def harvest_manuscript(pre: PreprocessedDocument) -> list[MentionCandidate]:
    """Extract MentionCandidate records from all prose spans in the document.

    Processes headings and paragraphs (anything with tokens in tokens_by_span).
    SceneBreak spans are not in tokens_by_span and are automatically skipped.

    Args:
        pre: A fully preprocessed document from the preprocessing stage.

    Returns:
        MentionCandidate records in document order (by span ordinal, then by
        token position within the span). Sentence-initial singletons are
        suppressed before returning.
    """
    sentence_initial_chars: set[int] = {
        s.tokens[0].start_char
        for s in pre.sentences
        if s.tokens
    }

    raw: list[MentionCandidate] = []
    for span_ordinal in sorted(pre.tokens_by_span.keys()):
        tokens = pre.tokens_by_span[span_ordinal]
        if not tokens:
            continue
        raw.extend(_extract_from_span(pre.source.path, tokens, span_ordinal))

    return _suppress_sentence_initial_singletons(raw, sentence_initial_chars)
