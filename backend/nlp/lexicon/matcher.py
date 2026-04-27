"""
Lexicon matcher - compiles BootstrappedLexiconEntry records into an
Aho-Corasick automaton and runs phrase matching over span text.

Matching operates on the normalised span text (after 1:1 character
substitutions) so that typographic variants are handled transparently.
Because the normalisation is 1:1, match offsets into the normalised text are
identical to offsets into the original text, and can be promoted directly to
absolute document positions by adding the span's start_char.

.. code-block:: mermaid

    flowchart TD
        A[list of BootstrappedLexiconEntry] --> B[Normalise each phrase\n1:1 substitutions]
        B --> C[Build Aho-Corasick automaton\none pattern per unique phrase]
        C --> D[For each span: normalise text\niterate automaton matches]
        D --> E{Word-boundary check\nbefore and after match}
        E -->|Pass| F[Emit MentionCandidate\nwith absolute offsets]
        E -->|Fail| G[Discard match]
        F --> H[list of MentionCandidate]
"""

from __future__ import annotations

import ahocorasick

from backend.nlp.parsing.preprocessing import _normalize_span_text
from backend.nlp.types import (
    BootstrappedLexiconEntry,
    MentionCandidate,
    SpanAnchor,
)
from backend.nlp.harvesting.shared import stable_hash_id


def compile_automaton(entries: list[BootstrappedLexiconEntry]) -> ahocorasick.Automaton:
    """Build an Aho-Corasick automaton from a list of lexicon entries.

    Each entry's phrase is normalised via the same 1:1 substitution used in
    preprocessing, so patterns match the normalised form of the span text.
    Duplicate normalised patterns are silently deduplicated; the first entry
    wins.

    Args:
        entries: BootstrappedLexiconEntry records from the induction stage.

    Returns:
        A compiled Aho-Corasick automaton. If entries is empty, an empty
        automaton is returned (iterating it yields nothing).
    """
    automaton: ahocorasick.Automaton = ahocorasick.Automaton()
    seen: set[str] = set()

    for entry in entries:
        normalised_pattern = _normalize_span_text(entry.phrase)
        if not normalised_pattern or normalised_pattern in seen:
            continue
        seen.add(normalised_pattern)
        # Store (normalized_phrase, pattern_length) as the payload so the
        # match loop can reconstruct the start index and look up the cluster key.
        automaton.add_word(normalised_pattern, (entry.normalized_phrase, len(normalised_pattern)))

    if len(automaton) > 0:
        automaton.make_automaton()

    return automaton


def match_text(
    automaton: ahocorasick.Automaton,
    original: str,
    path: str,
    span_ordinal: int,
    span_start: int,
) -> list[MentionCandidate]:
    """Run the automaton over one span and emit MentionCandidate records.

    Offsets are absolute document positions: span_start is added to the
    span-relative match position before storing in the anchor. This means the
    caller can slice doc.raw_text[anchor.start_char:anchor.end_char] to recover
    the exact matched surface form.

    Word-boundary checking rejects matches where the character immediately
    before or after the match is alphabetic or an apostrophe. This prevents
    'Aldous' from matching inside 'McAldous' or inside 'Aldous's'.

    Args:
        automaton: A compiled automaton from compile_automaton.
        original: The un-normalised span text (e.g. Heading.text or Paragraph.text).
        path: Document path for anchor construction.
        span_ordinal: Ordinal of the span being matched.
        span_start: Absolute character position of the first character of the
            span within the document raw_text.

    Returns:
        MentionCandidate records for all boundary-valid matches, in text order.
    """
    if len(automaton) == 0:
        return []

    normalised = _normalize_span_text(original)
    candidates: list[MentionCandidate] = []

    for end_idx, (normalized_phrase, phrase_len) in automaton.iter(normalised):
        start_idx = end_idx - phrase_len + 1

        # Reject matches where the preceding character is alphabetic or an
        # apostrophe (would indicate the match is a suffix of a longer word).
        if start_idx > 0 and (normalised[start_idx - 1].isalpha() or normalised[start_idx - 1] == "'"):
            continue

        # Reject matches where the following character is alphabetic or an
        # apostrophe (would indicate the match is a prefix of a longer word
        # or a possessive form whose base should not generate a separate match).
        if end_idx + 1 < len(normalised) and (normalised[end_idx + 1].isalpha() or normalised[end_idx + 1] == "'"):
            continue

        abs_start = span_start + start_idx
        abs_end = span_start + end_idx + 1
        surface = original[start_idx:end_idx + 1]

        candidates.append(MentionCandidate(
            surface=surface,
            normalized=normalized_phrase,
            anchor=SpanAnchor(
                path=path,
                span_ordinal=span_ordinal,
                start_char=abs_start,
                end_char=abs_end,
            ),
            has_title_prefix=False,
            has_possessive=False,
            has_location_context=False,
            rule_source='lexicon',
            candidate_id=stable_hash_id(path, str(span_ordinal), surface),
        ))

    return candidates
