"""
Tests for backend/nlp/parsing/preprocessing.py.

Each test encodes a non-obvious decision or invariant. Tests that verify
obvious behaviour (e.g. "a word becomes a token") are only included when
the specific decision about how to handle that behaviour is non-trivial.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess, _normalize_span_text


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make(text: str, path: str = "doc.md"):
    doc = parse(path, text)
    return doc, preprocess(doc)


# ---------------------------------------------------------------------------
# Offset invariant
# ---------------------------------------------------------------------------

class TestOffsets:
    def test_token_offsets_point_into_raw_text(self):
        # The core invariant: token offsets are absolute positions in the raw
        # document text. If any normalization step produces a string shorter
        # than the original, offsets shift silently and slices return wrong text.
        text = "She said: hello world."
        doc, pre = make(text)
        for tokens in pre.tokens_by_span.values():
            for token in tokens:
                assert text[token.start_char:token.end_char] == token.raw_text

    def test_non_ascii_normalization_preserves_offsets(self):
        # Typographic characters (curly quotes, em-dashes) are normalised 1:1.
        # If any substitution were 2:1 (two source chars to one output char),
        # every token after that point would be shifted. This test places a
        # curly-quote paragraph before a second paragraph to catch that shift.
        text = "“she said”.\n\nNext paragraph word."
        doc, pre = make(text)
        for tokens in pre.tokens_by_span.values():
            for token in tokens:
                assert text[token.start_char:token.end_char] == token.raw_text

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=300)
    def test_token_offset_invariant_holds_for_arbitrary_text(self, text: str):
        # Hypothesis exercises the offset invariant across a wide range of
        # Unicode input. A systematic offset bug (e.g. byte vs code-point
        # counting) would surface reliably even though ASCII examples pass.
        doc, pre = make(text)
        for tokens in pre.tokens_by_span.values():
            for token in tokens:
                assert text[token.start_char:token.end_char] == token.raw_text


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_curly_quotes_become_straight(self):
        # Curly quotes in the raw text must be converted to straight ASCII
        # quotes so the quote-detection logic (which matches on `"`) can pair
        # them. Without this, typographic quotes produce no QuoteSpan records.
        text = "“Hello.”"
        doc, pre = make(text)
        tokens = pre.tokens_by_span[doc.paragraphs[0].span_ordinal]
        normalized_surfaces = [t.text for t in tokens]
        assert '"' in normalized_surfaces

    def test_em_dash_becomes_hyphen(self):
        # Em-dashes are substituted so downstream matchers that look for
        # hyphenated ranges do not need to handle both U+2014 and '-'.
        text = "A—B"
        doc, pre = make(text)
        tokens = pre.tokens_by_span[doc.paragraphs[0].span_ordinal]
        assert any(t.text == '-' for t in tokens)

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=200)
    def test_normalization_is_idempotent(self, text: str):
        # Normalising twice must produce the same result as normalising once.
        # If any substitution introduced a character that a later substitution
        # would also replace, the pipeline would need to track normalisation
        # rounds, which the current design intentionally avoids.
        once = _normalize_span_text(text)
        twice = _normalize_span_text(once)
        assert once == twice

    def test_normalization_preserves_length(self):
        # Every substitution is 1:1, so the normalised string must be the same
        # length as the original. A shorter normalised string would shift all
        # token offsets computed from that point onward.
        text = "“Hello” — world…"
        normalised = _normalize_span_text(text)
        assert len(normalised) == len(text)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestTokenization:
    def test_contraction_is_one_token(self):
        # The token regex includes `(?:'\w+)*` so that contractions like
        # "don't" and "she'd" become single tokens. Without this, a contraction
        # would split into three tokens (word, apostrophe, word), breaking
        # entity matching for character names like "O'Brien".
        text = "She don't know O'Brien."
        doc, pre = make(text)
        tokens = pre.tokens_by_span[doc.paragraphs[0].span_ordinal]
        surfaces = [t.text for t in tokens]
        assert "don't" in surfaces
        assert "O'Brien" in surfaces

    def test_punctuation_is_a_single_token(self):
        # Single non-whitespace characters that are not part of a word must
        # each be their own token so sentence splitting and quote detection can
        # find them by exact text match.
        text = "A, B; C."
        doc, pre = make(text)
        tokens = pre.tokens_by_span[doc.paragraphs[0].span_ordinal]
        surfaces = [t.text for t in tokens]
        assert ',' in surfaces
        assert ';' in surfaces

    def test_heading_tokens_present_in_tokens_by_span(self):
        # Headings are content spans and must be tokenised. A bug that skipped
        # headings would silently produce no tokens for heading spans, causing
        # harvesters that iterate tokens_by_span to miss heading content.
        text = "# The Long Night"
        doc, pre = make(text)
        assert doc.headings[0].span_ordinal in pre.tokens_by_span

    def test_scene_break_not_in_tokens_by_span(self):
        # SceneBreak spans carry no text, so they must not appear in
        # tokens_by_span. A bug that attempted to tokenise a SceneBreak would
        # either crash (SceneBreak has no .text field) or produce an empty
        # entry that confuses harvesters expecting only Heading and Paragraph
        # ordinals as keys.
        text = "Before.\n---\nAfter."
        doc, pre = make(text)
        sb_ordinal = doc.scene_breaks[0].span_ordinal
        assert sb_ordinal not in pre.tokens_by_span


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    def test_period_uppercase_is_boundary(self):
        # A period followed by an uppercase word is the primary sentence
        # boundary signal. This is the case where splitting must happen.
        text = "She ran. He followed."
        doc, pre = make(text)
        assert len(pre.sentences) == 2

    def test_abbreviation_period_is_not_boundary(self):
        # A period after a known title abbreviation (Dr, Mr, etc.) must not
        # be treated as a sentence boundary. Without this rule, "Dr. Smith"
        # splits into a one-token sentence "Dr." and the remainder, which
        # corrupts entity mention windows that depend on sentence context.
        text = "Dr. Smith examined the patient."
        doc, pre = make(text)
        assert len(pre.sentences) == 1

    def test_period_lowercase_continuation_is_not_boundary(self):
        # A period followed by a lowercase word is not a sentence boundary.
        # This covers abbreviations not in _ABBREV_TITLES (e.g. "etc.") and
        # mid-sentence decimal numbers, without needing an exhaustive list.
        text = "Add salt etc. and stir."
        doc, pre = make(text)
        assert len(pre.sentences) == 1

    def test_sentences_ordered_across_spans(self):
        # Sentences from multiple spans are appended in document order to
        # pre.sentences. A bug that extended them in the wrong order would
        # mismatch sentence context with token position in the retrieval stage.
        text = "Para one. Para one sentence two.\n\nPara two."
        doc, pre = make(text)
        starts = [s.start_char for s in pre.sentences]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Quote detection
# ---------------------------------------------------------------------------

class TestQuoteDetection:
    def test_matched_quotes_produce_quote_span(self):
        # The basic contract: a pair of `"` tokens around text produces one
        # QuoteSpan. This is the core case quote detection exists to handle.
        text = 'She said "hello there" to him.'
        doc, pre = make(text)
        assert len(pre.quote_spans) == 1

    def test_quote_inner_text_excludes_marks(self):
        # inner_text must be the content between the marks, not including the
        # marks themselves. If the slice were raw_text[open.start:close.end]
        # instead of raw_text[open.end:close.start], inner_text would silently
        # include the quotation characters, corrupting dialogue extraction.
        text = 'She said "hello" to him.'
        doc, pre = make(text)
        assert pre.quote_spans[0].inner_text == "hello"

    def test_unmatched_quote_is_silently_discarded(self):
        # An odd number of `"` characters is common in prose (em-dash before
        # closing quotes, nested quotation conventions, etc.). Raising an error
        # would halt preprocessing on real manuscripts. Discarding the unmatched
        # opener is the correct silent recovery.
        text = 'He said "wait.'
        doc, pre = make(text)
        assert len(pre.quote_spans) == 0

    def test_typographic_quotes_are_detected(self):
        # Curly quotes (“ / ”) are normalised to straight ASCII `"`
        # before tokenisation. Without this, typographic quotes would not match
        # the `"` check in _detect_quotes and would silently produce zero
        # QuoteSpan records for documents that use curly quotes (most word
        # processors produce these by default).
        text = "“Hello.”"
        doc, pre = make(text)
        assert len(pre.quote_spans) == 1

    def test_quote_offsets_point_into_raw_text(self):
        # QuoteSpan start_char and end_char must be absolute positions in the
        # raw document text, consistent with all other offset fields. A bug
        # that used span-relative positions instead would produce anchors that
        # silently point to the wrong passage.
        text = 'Start. She said "hello" end.'
        doc, pre = make(text)
        qs = pre.quote_spans[0]
        assert text[qs.start_char:qs.end_char] == '"hello"'

    def test_single_quoted_dialogue_detected(self):
        # British-style dialogue and internal character speech use single quotes.
        # Without single-quote detection, these passages produce no QuoteSpan
        # records and the attribution stage silently misses all dialogue in
        # single-quote manuscripts.
        text = "'Hello there,' she said."
        doc, pre = make(text)
        assert len(pre.quote_spans) == 1
        assert pre.quote_spans[0].inner_text == "Hello there,"

    def test_apostrophe_not_confused_with_single_quote(self):
        # Apostrophes inside contractions and possessives (don't, Aldous's) are
        # absorbed by the tokenizer into a single word token and must not appear
        # as standalone quote delimiters. Without this invariant, contractions in
        # the vicinity of single-quote dialogue would corrupt the open/close pairing.
        text = "She didn't hear him. 'Go away,' he said."
        doc, pre = make(text)
        # Only one QuoteSpan: the single-quote dialogue. The apostrophe in
        # "didn't" is absorbed into the contraction token and is invisible to
        # the quote detector.
        assert len(pre.quote_spans) == 1
        assert "away" in pre.quote_spans[0].inner_text

    def test_plural_possessive_apostrophe_not_confused_with_single_quote(self):
        # Plural possessives ending with a bare apostrophe ("James'", "soldiers'")
        # produce a standalone ' token because the tokenizer has no following word
        # character to absorb. Without the preceding-character boundary check in
        # _pair_quote_tokens, this token pairs with the next real quote delimiter
        # and produces a false QuoteSpan that spans the possessive word and the
        # surrounding prose rather than the actual dialogue.
        text = "James' sword fell. 'Hello,' she said."
        doc, pre = make(text)
        assert len(pre.quote_spans) == 1
        assert "Hello" in pre.quote_spans[0].inner_text

    def test_typographic_single_quotes_are_detected(self):
        # Curly single quotes (‘ / ’) are normalised to ASCII `'`
        # before tokenisation. Without this, typographic single quotes would
        # not produce QuoteSpan records for manuscripts from word processors that
        # auto-convert to curly quotes.
        text = "‘Hello.’"
        doc, pre = make(text)
        assert len(pre.quote_spans) == 1


# ---------------------------------------------------------------------------
# Structural markers
# ---------------------------------------------------------------------------

class TestStructuralMarkers:
    def test_heading_produces_structural_marker(self):
        # Harvesters use structural_markers to identify heading context without
        # re-scanning the span type. If headings were omitted, a harvester
        # looking for heading spans in structural_markers would silently find none.
        text = "# Chapter One\nBody."
        doc, pre = make(text)
        kinds = [m.kind.value for m in pre.structural_markers]
        assert 'heading' in kinds

    def test_scene_break_produces_structural_marker(self):
        # Scene breaks must appear in structural_markers so harvesters can
        # detect them without inspecting the separate scene_breaks list on
        # the parsed document.
        text = "Before.\n---\nAfter."
        doc, pre = make(text)
        kinds = [m.kind.value for m in pre.structural_markers]
        assert 'scene_break' in kinds

    def test_paragraph_does_not_produce_structural_marker(self):
        # Paragraphs are the default content type. Including them in
        # structural_markers would make the list useless as a fast filter
        # for non-paragraph spans.
        text = "Just a paragraph."
        doc, pre = make(text)
        assert len(pre.structural_markers) == 0
