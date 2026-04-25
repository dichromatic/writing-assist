"""
Tests for backend/nlp/types.py.

Each test encodes a non-obvious design decision whose breakage would be
invisible in the pipeline. Tests that would only verify Python language
behaviour (SHA-256 determinism, dataclass equality, enum string values)
are omitted.
"""

import pytest
from backend.nlp.types import (
    DocumentAnchor,
    SectionAnchor,
    SpanAnchor,
    stable_hash_id,
)


class TestStableHashId:
    def test_no_cross_component_collision(self):
        # The null-byte separator prevents ("ab", "c") from hashing
        # identically to ("a", "bc"). Without the separator, naive string
        # concatenation would make these indistinguishable.
        assert stable_hash_id("ab", "c") != stable_hash_id("a", "bc")

    def test_returns_16_char_hex_string(self):
        # Locks in the output format that all candidate_id, cluster_id, and
        # entry_id fields depend on. A silent change in truncation length would
        # break stored IDs without an obvious failure.
        result = stable_hash_id("anything")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)


class TestAnchors:
    def test_document_anchor_is_frozen(self):
        # Anchors are frozen so they can be used as dict keys throughout the
        # pipeline. A mutable anchor would fail silently as a key: it could
        # be inserted but never retrieved after mutation.
        anchor = DocumentAnchor(path="doc.md")
        with pytest.raises((AttributeError, TypeError)):
            anchor.path = "other.md"  # type: ignore[misc]

    def test_span_anchor_is_frozen(self):
        anchor = SpanAnchor(path="doc.md", span_ordinal=0, start_char=0, end_char=10)
        with pytest.raises((AttributeError, TypeError)):
            anchor.span_ordinal = 1  # type: ignore[misc]

    def test_section_anchor_is_frozen(self):
        anchor = SectionAnchor(path="doc.md", section_index=0)
        with pytest.raises((AttributeError, TypeError)):
            anchor.section_index = 1  # type: ignore[misc]

    def test_span_anchor_usable_as_dict_key(self):
        # The actual requirement behind frozen: anchors must work as dict keys
        # so tokens_by_span and similar structures can index by anchor.
        anchor = SpanAnchor(path="doc.md", span_ordinal=0, start_char=0, end_char=5)
        d = {anchor: "value"}
        assert d[anchor] == "value"

    def test_section_anchor_usable_as_dict_key(self):
        anchor = SectionAnchor(path="doc.md", section_index=1)
        d = {anchor: "section_data"}
        assert d[anchor] == "section_data"
