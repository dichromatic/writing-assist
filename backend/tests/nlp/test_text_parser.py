"""
Tests for backend/nlp/parsing/text_parser.py and document_parser.py.

These tests lock the structure heuristics that matter for loose text notes.
Without them, a future refactor could silently collapse dossier titles and
outline headings back into paragraphs while still leaving the downstream
pipeline superficially functional.
"""

from backend.nlp.parsing.document_parser import parse as parse_document
from backend.nlp.parsing.text_parser import parse as parse_text


class TestTextParser:
    def test_uppercase_banner_line_becomes_heading(self):
        # Crew summary files use uppercase banner lines to define the subject
        # of the following block. If those lines fall back to paragraphs, the
        # structured-note pipeline loses the strongest section boundary it has.
        text = "RADIANT ESTUARY - PRIMARY BRIDGE CREW SUMMARY\n\nBody text."
        doc = parse_text("crew.txt", text)

        assert len(doc.headings) == 1
        assert doc.headings[0].normalized_text == "RADIANT ESTUARY - PRIMARY BRIDGE CREW SUMMARY"
        assert len(doc.sections) == 1
        assert doc.sections[0].heading is not None

    def test_numbered_outline_line_becomes_heading(self):
        # Story-planning notes use numbered beat headings like "0.1 - ...".
        # Treating them as headings preserves outline structure for later
        # routing instead of flattening the plan into plain prose.
        text = "0.1 - Last Admiralty Session\n\n- Beat one."
        doc = parse_text("plan.txt", text)

        assert len(doc.headings) == 1
        assert doc.headings[0].normalized_text == "0.1 - Last Admiralty Session"
        assert doc.headings[0].level == 3

    def test_label_value_line_stays_paragraph(self):
        # Planning notes also contain metadata labels like "Tone: ...".
        # Promoting those to headings would fragment the document into noisy
        # pseudo-sections and drown out the meaningful outline headings.
        text = "Tone: heavy -> hopeful -> anticipatory\n\nBody text."
        doc = parse_text("plan.txt", text)

        assert len(doc.headings) == 0
        assert len(doc.paragraphs) == 2


class TestDocumentParser:
    def test_txt_suffix_dispatches_to_text_parser(self):
        # The dispatcher must route .txt files through the text heuristics.
        # If it falls back to Markdown-only parsing, uppercase dossier titles
        # disappear silently because the old parser sees only paragraphs.
        text = "RADIANT ESTUARY - PRIMARY BRIDGE CREW SUMMARY\n\nBody text."
        doc = parse_document("crew.txt", text)

        assert len(doc.headings) == 1
