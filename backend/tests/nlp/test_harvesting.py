"""
Tests for backend/nlp/harvesting/shared.py and
        backend/nlp/harvesting/manuscript.py.

Each test encodes a non-obvious decision or invariant. The test comments
explain which decision is being locked in and why its breakage would be silent.
"""

import pytest

from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.harvesting.manuscript import harvest_manuscript
from backend.nlp.harvesting.shared import normalize_surface, is_stopword, TITLE_PREFIXES


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def pipeline(text: str, path: str = "doc.md"):
    """Run the full parse -> preprocess -> harvest pipeline on `text`."""
    doc = parse(path, text)
    pre = preprocess(doc)
    return harvest_manuscript(pre)


def normalized_keys(candidates):
    return {c.normalized for c in candidates}


# ---------------------------------------------------------------------------
# shared.py: normalize_surface
# ---------------------------------------------------------------------------

class TestNormalizeSurface:
    def test_strips_title_prefix(self):
        # normalize_surface drives the clustering key. If the title prefix is
        # not stripped, "Captain Aldous" and "Aldous" cluster under different
        # keys and the same character appears as two separate entities.
        assert normalize_surface("Captain Aldous") == "aldous"

    def test_strips_possessive_suffix(self):
        # "Aldous's" must cluster with "Aldous". Without stripping the
        # possessive, every character with at least one possessive occurrence
        # becomes a phantom entity distinct from their base form.
        assert normalize_surface("Aldous's") == "aldous"

    def test_strips_both_prefix_and_possessive(self):
        # "Captain Aldous's" must collapse to "aldous" - both transformations
        # apply in order: prefix stripped first, then possessive.
        assert normalize_surface("Captain Aldous's") == "aldous"

    def test_lowercase_only(self):
        # A bare name with no prefix or possessive is just lowercased.
        assert normalize_surface("Aldous") == "aldous"

    def test_title_only_falls_back_to_lowercase_title(self):
        # If the surface is only a title word with no name ("Captain" alone),
        # stripping the prefix leaves nothing; normalize returns the lowercased
        # original rather than an empty string. This prevents empty clustering
        # keys from colliding all such degenerate candidates.
        result = normalize_surface("Captain")
        assert result == "captain"


# ---------------------------------------------------------------------------
# shared.py: is_stopword
# ---------------------------------------------------------------------------

class TestIsStopword:
    def test_common_word_is_stopword(self):
        # "The" at the start of a sentence must be filtered, not treated as
        # a character name. The check must be case-insensitive since stopwords
        # are stored lowercase but sentence-initial tokens are capitalised.
        assert is_stopword("The") is True
        assert is_stopword("the") is True

    def test_proper_name_is_not_stopword(self):
        # Character names must not be suppressed by the stopword filter.
        assert is_stopword("Aldous") is False


# ---------------------------------------------------------------------------
# Candidate extraction: title-prefix rule
# ---------------------------------------------------------------------------

class TestTitlePrefixExtraction:
    def test_title_plus_name_produces_candidate(self):
        # The primary reason title-prefix extraction exists: "Captain Aldous"
        # is strong evidence of a character name regardless of position.
        candidates = pipeline("Captain Aldous arrived.")
        assert any(c.surface == "Captain Aldous" for c in candidates)

    def test_abbreviated_title_plus_name(self):
        # "Dr. Smith" has a period between the title and the name. The period
        # must be skipped when collecting the name tokens, or the phrase stops
        # at "Dr" and "Smith" falls through as a bare capitalized name instead.
        candidates = pipeline("Dr. Smith examined the patient.")
        assert any("Smith" in c.surface for c in candidates)
        title_candidate = next(c for c in candidates if "Smith" in c.surface)
        assert title_candidate.has_title_prefix is True

    def test_bare_title_alone_does_not_produce_title_prefix_candidate(self):
        # A title token with no following name word ("Captain said nothing.")
        # must not produce a title_prefix candidate. The title alone names no
        # entity and would create a spurious cluster key "captain".
        candidates = pipeline("Captain said nothing.")
        title_cands = [c for c in candidates if c.rule_source == 'title_prefix']
        assert len(title_cands) == 0

    def test_title_prefix_candidate_has_flag_set(self):
        # The has_title_prefix flag must be True for title-rule candidates.
        # Downstream promotion uses this flag to boost confidence.
        candidates = pipeline("Lord Vayne walked in.")
        cand = next((c for c in candidates if c.rule_source == 'title_prefix'), None)
        assert cand is not None
        assert cand.has_title_prefix is True

    def test_name_tokens_not_duplicated_as_bare_capitalized(self):
        # After "Captain Aldous" is extracted, "Aldous" must not also appear
        # as a separate bare_capitalized candidate from the same span. The
        # consumed-token mechanism prevents this double extraction.
        candidates = pipeline("Captain Aldous sat down.")
        bare = [c for c in candidates if c.rule_source == 'bare_capitalized' and c.surface == 'Aldous']
        assert len(bare) == 0


# ---------------------------------------------------------------------------
# Candidate extraction: possessive rule
# ---------------------------------------------------------------------------

class TestPossessiveExtraction:
    def test_possessive_produces_candidate(self):
        # "Aldous's" must be detected and produce a candidate. The possessive
        # form is strong evidence that the base ("Aldous") is an entity.
        candidates = pipeline("Aldous's sword was missing.")
        assert any(c.rule_source == 'possessive' for c in candidates)

    def test_possessive_normalized_form_strips_suffix(self):
        # The normalised key for "Aldous's" must be "aldous" so it clusters
        # with bare "Aldous" occurrences. If the suffix is not stripped, every
        # character with possessive usage becomes a separate phantom entity.
        candidates = pipeline("Aldous's sword was missing.")
        poss = next(c for c in candidates if c.rule_source == 'possessive')
        assert poss.normalized == "aldous"

    def test_possessive_has_flag_set(self):
        # has_possessive must be True on possessive candidates. Promotion uses
        # this flag to recognise that the cluster has been seen in possessive
        # context, which is a reliable indicator of personhood.
        candidates = pipeline("Rhea's decision surprised them.")
        poss = next((c for c in candidates if c.rule_source == 'possessive'), None)
        assert poss is not None
        assert poss.has_possessive is True

    def test_possessive_and_bare_cluster_together(self):
        # "Aldous's" and "Aldous" both normalise to "aldous". This test
        # verifies that both forms produce candidates with the same normalised
        # key, which is the pre-condition for correct clustering in phase P.5.
        candidates = pipeline("Aldous came in. Aldous's sword fell.")
        keys = normalized_keys(candidates)
        assert "aldous" in keys
        assert len([c for c in candidates if c.normalized == "aldous"]) >= 2


# ---------------------------------------------------------------------------
# Candidate extraction: bare capitalised names
# ---------------------------------------------------------------------------

class TestBareCapitalized:
    def test_recurring_capitalized_name_produces_candidate(self):
        # A name that appears mid-sentence (not sentence-initial) is direct
        # evidence: it's capitalised by convention, not by position.
        candidates = pipeline("She greeted Aldous warmly. Aldous smiled.")
        bare = [c for c in candidates if c.normalized == "aldous"]
        assert len(bare) >= 1

    def test_stopword_filtered_out(self):
        # Sentence-initial "The" is a stopword and must produce no candidate.
        # Without the stopword filter, every common function word at sentence
        # start would become a spurious entity candidate.
        candidates = pipeline("The door opened.")
        assert not any(c.normalized == "the" for c in candidates)

    def test_sentence_initial_singleton_suppressed(self):
        # A capitalised word that appears only once and only at sentence start
        # ("Sunlight flooded the room.") provides no evidence of a proper name.
        # It must be suppressed to avoid adding common nouns to the entity pool.
        candidates = pipeline("Sunlight flooded the room.")
        assert not any(c.normalized == "sunlight" for c in candidates)

    def test_recurring_sentence_initial_name_not_suppressed(self):
        # If the same normalised key appears more than once (even if some
        # occurrences are sentence-initial), all occurrences are kept.
        # The recurrence itself is the evidence.
        candidates = pipeline("Aldous entered. Aldous sat down.")
        aldous_cands = [c for c in candidates if c.normalized == "aldous"]
        assert len(aldous_cands) >= 2

    def test_name_with_support_in_other_span_not_suppressed(self):
        # A name that appears sentence-initially in one span but mid-sentence
        # in another span must not be suppressed. The mid-sentence occurrence
        # provides the support needed to survive suppression.
        candidates = pipeline("Rhea sat.\n\nShe greeted Rhea warmly.")
        rhea_cands = [c for c in candidates if c.normalized == "rhea"]
        assert len(rhea_cands) >= 1


# ---------------------------------------------------------------------------
# Anchor and provenance
# ---------------------------------------------------------------------------

class TestAnchorsAndProvenance:
    def test_every_candidate_carries_document_path(self):
        # Source attribution is a pipeline invariant. If any candidate has a
        # wrong or empty path, retrieval will point to the wrong document.
        candidates = pipeline("Captain Aldous arrived.", path="ch01.md")
        assert all(c.anchor.path == "ch01.md" for c in candidates)

    def test_candidate_anchor_start_points_into_raw_text(self):
        # The anchor start_char must be a valid offset into the document
        # raw_text. A wrong offset would silently return the wrong passage
        # when a harvested candidate is looked up in the retrieval stage.
        text = "She greeted Aldous warmly."
        doc = parse("doc.md", text)
        pre = preprocess(doc)
        candidates = harvest_manuscript(pre)
        for c in candidates:
            assert text[c.anchor.start_char:c.anchor.end_char] == c.surface

    def test_candidate_id_is_deterministic(self):
        # candidate_id is used to deduplicate records across pipeline runs.
        # Running the harvester twice on the same document must produce the
        # same IDs in the same order, or deduplication silently fails.
        text = "Captain Aldous arrived."
        c1 = pipeline(text)
        c2 = pipeline(text)
        assert [c.candidate_id for c in c1] == [c.candidate_id for c in c2]
