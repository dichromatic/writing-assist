"""
Tests for backend/nlp/parsing/markdown_parser.py.

Each test encodes a non-obvious parsing decision or offset invariant. Tests
that verify obvious Markdown syntax (e.g. "# produces a heading") are only
included when the decision about what to do with that syntax is non-trivial.
"""

import pytest
from backend.nlp.parsing.markdown_parser import parse


# ---------------------------------------------------------------------------
# Offset correctness
# ---------------------------------------------------------------------------

class TestOffsets:
    def test_heading_offsets_point_into_raw_text(self):
        # Offsets must index into raw_text correctly so anchors built from
        # them can retrieve the original text without re-scanning.
        text = "# Chapter One\nSome prose."
        doc = parse("doc.md", text)
        h = doc.headings[0]
        assert text[h.start_char:h.end_char] == h.text

    def test_paragraph_offsets_point_into_raw_text(self):
        text = "# Heading\nFirst paragraph.\n\nSecond paragraph."
        doc = parse("doc.md", text)
        for para in doc.paragraphs:
            assert text[para.start_char:para.end_char] == para.text

    def test_scene_break_offsets_point_into_raw_text(self):
        text = "Before.\n---\nAfter."
        doc = parse("doc.md", text)
        sb = doc.scene_breaks[0]
        assert text[sb.start_char:sb.end_char] == "---"

    def test_non_ascii_offsets_are_code_point_positions(self):
        # Offsets are Python string indices (Unicode code points), not byte
        # positions. A multi-byte character must not shift the offset of the
        # span that follows it.
        text = "Café is good.\n# Next"
        doc = parse("doc.md", text)
        h = doc.headings[0]
        assert text[h.start_char:h.end_char] == h.text

    def test_span_ordinals_are_strictly_increasing(self):
        # Ordinals must be assigned in document order. A break in the sequence
        # would corrupt the position-based lookup used by the retrieval stage.
        text = "# H1\nPara one.\n---\n# H2\nPara two."
        doc = parse("doc.md", text)
        all_ordinals = sorted(
            [h.span_ordinal for h in doc.headings]
            + [p.span_ordinal for p in doc.paragraphs]
            + [s.span_ordinal for s in doc.scene_breaks]
        )
        assert all_ordinals == list(range(len(all_ordinals)))


# ---------------------------------------------------------------------------
# Normalized text
# ---------------------------------------------------------------------------

class TestNormalizedText:
    def test_normalized_text_collapses_internal_newlines(self):
        # A hard-wrapped paragraph has internal newlines in its raw text.
        # normalized_text converts them to spaces for retrieval.
        text = "Line one\nline two\nline three."
        doc = parse("doc.md", text)
        assert doc.paragraphs[0].normalized_text == "Line one line two line three."

    def test_normalized_text_does_not_mutate_raw_text(self):
        # raw_text must be the unmodified source. If normalized_text and text
        # point to the same object, a later mutation would corrupt anchors.
        text = "Some   spaced   words."
        doc = parse("doc.md", text)
        para = doc.paragraphs[0]
        assert para.text == "Some   spaced   words."
        assert para.normalized_text == "Some spaced words."

    def test_heading_normalized_text_strips_hashes(self):
        # Downstream stages that match on heading text should not see # markers.
        text = "# Chapter One"
        doc = parse("doc.md", text)
        assert doc.headings[0].normalized_text == "Chapter One"

    def test_heading_normalized_text_strips_closing_hashes(self):
        # ATX headings may have a trailing closing sequence (e.g. "# Title #").
        # normalized_text must strip it; otherwise lexicon entries would
        # include trailing # as part of the heading surface.
        text = "# Chapter One ##"
        doc = parse("doc.md", text)
        assert doc.headings[0].normalized_text == "Chapter One"


# ---------------------------------------------------------------------------
# Section derivation
# ---------------------------------------------------------------------------

class TestSections:
    def test_document_with_no_headings_produces_one_section_with_no_heading(self):
        # A document without headings is a single section. heading=None signals
        # to harvesters that no heading context is available for field detection.
        text = "Just a paragraph."
        doc = parse("doc.md", text)
        assert len(doc.sections) == 1
        assert doc.sections[0].heading is None

    def test_heading_starts_a_new_section(self):
        text = "Intro.\n# Chapter\nBody."
        doc = parse("doc.md", text)
        # Section 0: pre-heading content. Section 1: Chapter and its body.
        assert len(doc.sections) == 2
        assert doc.sections[0].heading is None
        assert doc.sections[1].heading is not None
        assert doc.sections[1].heading.normalized_text == "Chapter"

    def test_document_starting_with_heading_has_no_empty_section_zero(self):
        # When a document starts with a heading there is no content before it,
        # so section 0 must not be created as an empty placeholder.
        text = "# Chapter\nBody."
        doc = parse("doc.md", text)
        assert len(doc.sections) == 1
        assert doc.sections[0].heading is not None

    def test_section_span_ordinals_contain_all_spans_in_that_section(self):
        text = "# H\nPara one.\nPara two."
        doc = parse("doc.md", text)
        section = doc.sections[0]
        all_doc_ordinals = (
            {h.span_ordinal for h in doc.headings}
            | {p.span_ordinal for p in doc.paragraphs}
        )
        assert set(section.span_ordinals) == all_doc_ordinals

    def test_empty_document_produces_one_empty_section(self):
        doc = parse("doc.md", "")
        assert len(doc.sections) == 1
        assert doc.sections[0].span_ordinals == []


# ---------------------------------------------------------------------------
# Scene derivation
# ---------------------------------------------------------------------------

class TestScenes:
    def test_document_with_no_scene_breaks_is_one_scene(self):
        text = "Para one.\n\nPara two."
        doc = parse("doc.md", text)
        assert len(doc.scenes) == 1

    def test_scene_break_produces_two_scenes(self):
        text = "Before.\n---\nAfter."
        doc = parse("doc.md", text)
        assert len(doc.scenes) == 2

    def test_scene_break_ordinal_not_in_any_scene_span_ordinals(self):
        # Scene breaks are boundary markers, not content. Including them in
        # span_ordinals would require every harvester to filter them out.
        text = "Before.\n---\nAfter."
        doc = parse("doc.md", text)
        sb_ordinal = doc.scene_breaks[0].span_ordinal
        for scene in doc.scenes:
            assert sb_ordinal not in scene.span_ordinals

    def test_scene_content_is_split_correctly_across_break(self):
        text = "Before.\n---\nAfter."
        doc = parse("doc.md", text)
        before_ordinal = doc.paragraphs[0].span_ordinal
        after_ordinal = doc.paragraphs[1].span_ordinal
        scene_0_ordinals = set(doc.scenes[0].span_ordinals)
        scene_1_ordinals = set(doc.scenes[1].span_ordinals)
        assert before_ordinal in scene_0_ordinals
        assert after_ordinal in scene_1_ordinals
        assert before_ordinal not in scene_1_ordinals
        assert after_ordinal not in scene_0_ordinals

    def test_asterisk_scene_break_produces_scene(self):
        # *** is a valid CommonMark thematic break and must be treated as a
        # scene break. Authors using asterisms (***) instead of dashes should
        # not have to convert their manuscripts for the parser to split scenes.
        text = "Before.\n***\nAfter."
        doc = parse("doc.md", text)
        assert len(doc.scene_breaks) == 1
        assert len(doc.scenes) == 2

    def test_underscore_scene_break_produces_scene(self):
        # ___ is the third CommonMark thematic break style. Including it means
        # the parser accepts all three standard styles without special-casing.
        text = "Before.\n___\nAfter."
        doc = parse("doc.md", text)
        assert len(doc.scene_breaks) == 1
        assert len(doc.scenes) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_lines_do_not_become_spans(self):
        text = "\n\n# Heading\n\n\nParagraph.\n\n"
        doc = parse("doc.md", text)
        assert len(doc.headings) == 1
        assert len(doc.paragraphs) == 1
        assert len(doc.scene_breaks) == 0

    def test_empty_heading_treated_as_paragraph(self):
        # A heading with no text after the # (e.g. "# ") provides no
        # structural information and must not create an anonymous section
        # boundary that would split content incorrectly.
        text = "# \nActual content."
        doc = parse("doc.md", text)
        assert len(doc.headings) == 0
        assert len(doc.paragraphs) == 1

    def test_hard_wrapped_paragraph_is_one_span(self):
        # Lines without a blank line between them form a single paragraph.
        # Splitting them into multiple spans would break retrieval windows
        # that depend on paragraph-level context.
        text = "Line one\nLine two\nLine three"
        doc = parse("doc.md", text)
        assert len(doc.paragraphs) == 1

    def test_raw_text_is_unchanged(self):
        text = "# Heading\n\nParagraph."
        doc = parse("doc.md", text)
        assert doc.raw_text is text

    def test_path_propagates_to_all_anchors(self):
        text = "# H\nP.\n---"
        doc = parse("custom/path.md", text)
        assert doc.headings[0].anchor.path == "custom/path.md"
        assert doc.paragraphs[0].anchor.path == "custom/path.md"
        assert doc.scene_breaks[0].anchor.path == "custom/path.md"
        assert doc.sections[0].anchor.path == "custom/path.md"
