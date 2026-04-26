"""
Tests for backend/nlp/lexicon/induction.py,
        backend/nlp/lexicon/matcher.py, and
        backend/nlp/lexicon/bootstrap.py.

Each test encodes a non-obvious decision or invariant. Comments explain
which decision is locked in and why its breakage would be silent.
"""

import pytest

from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.harvesting.manuscript import harvest_manuscript
from backend.nlp.clustering.clustering import cluster_mentions
from backend.nlp.clustering.linking import link_clusters
from backend.nlp.lexicon.induction import induce_lexicon
from backend.nlp.lexicon.matcher import compile_automaton, match_text
from backend.nlp.lexicon.bootstrap import bootstrap
from backend.nlp.types import LexiconCategory, SpanAnchor, stable_hash_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def harvest_and_cluster(text: str, path: str = "doc.md"):
    doc = parse(path, text)
    pre = preprocess(doc)
    candidates = harvest_manuscript(pre)
    clusters = cluster_mentions(candidates)
    link_clusters(clusters, [], [], [])
    return clusters


def make_entry(phrase: str, normalized: str, category=LexiconCategory.UNRESOLVED, path="doc.md"):
    """Build a minimal BootstrappedLexiconEntry for matcher tests."""
    from backend.nlp.types import BootstrappedLexiconEntry
    return BootstrappedLexiconEntry(
        phrase=phrase,
        normalized_phrase=normalized,
        category=category,
        anchors=[],
        occurrence_count=1,
        archetypes_seen=['manuscript'],
        rule_sources=['bare_capitalized'],
        induction_pass=0,
        entry_id=stable_hash_id(path, phrase),
    )


# ---------------------------------------------------------------------------
# Induction: threshold
# ---------------------------------------------------------------------------

class TestInductionThreshold:
    def test_high_occurrence_count_meets_threshold(self):
        # occurrence_count >= 2 must be sufficient for induction regardless of
        # title or possessive flags. A cluster seen twice is evidenced enough
        # to be a pattern candidate for the next pass.
        clusters = harvest_and_cluster("She greeted Aldous warmly. Aldous smiled.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        assert any(e.normalized_phrase == "aldous" for e in lexicon)

    def test_title_support_meets_threshold_with_count_one(self):
        # A titled name ("Captain Aldous") must be inducted even if it appears
        # only once. The title prefix is strong evidence regardless of frequency.
        clusters = harvest_and_cluster("She met Captain Aldous by the gate.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        assert any(e.normalized_phrase == "aldous" for e in lexicon)

    def test_possessive_support_meets_threshold_with_count_one(self):
        # A possessive form ("Aldous's") must be inducted even if it appears
        # only once. The possessive implies personhood independently of count.
        clusters = harvest_and_cluster("Aldous's sword was missing.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        assert any(e.normalized_phrase == "aldous" for e in lexicon)

    def test_bare_singleton_not_inducted(self):
        # A bare capitalized name that appears exactly once with no title or
        # possessive support must not be inducted. It provides no signal beyond
        # the capitalisation convention and would add noise to the lexicon.
        # "Sunlight" appears mid-sentence once - no title, no possessive.
        clusters = harvest_and_cluster("She saw Sunlight through the window.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        assert not any(e.normalized_phrase == "sunlight" for e in lexicon)

    def test_stopword_not_inducted_regardless_of_count(self):
        # Even if a stopword appears many times capitalized, it must never
        # become a lexicon entry. Without this guard, "The" or "He" could
        # become automaton patterns that produce false matches on every sentence.
        clusters = harvest_and_cluster("He arrived. He departed. He returned.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        assert not any(e.normalized_phrase == "he" for e in lexicon)


# ---------------------------------------------------------------------------
# Induction: category and entry structure
# ---------------------------------------------------------------------------

class TestInductionCategory:
    def test_titled_cluster_gets_character_category(self):
        # has_title_support is the reliable signal for personhood. A titled
        # cluster must receive LexiconCategory.CHARACTER, not UNRESOLVED. The
        # promotion stage uses the category to apply different scoring rules.
        clusters = harvest_and_cluster("She met Captain Aldous by the gate.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        aldous_entry = next(e for e in lexicon if e.normalized_phrase == "aldous")
        assert aldous_entry.category == LexiconCategory.CHARACTER

    def test_possessive_cluster_gets_character_category(self):
        # Possessive forms also reliably indicate personhood.
        clusters = harvest_and_cluster("Aldous's sword was missing.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        entry = next(e for e in lexicon if e.normalized_phrase == "aldous")
        assert entry.category == LexiconCategory.CHARACTER

    def test_bare_recurring_cluster_gets_unresolved_category(self):
        # A name that recurs but has no title or possessive signal cannot be
        # classified reliably as CHARACTER vs PLACE vs other. UNRESOLVED
        # prevents premature category assignment that would be wrong for half
        # the entities in a typical manuscript.
        clusters = harvest_and_cluster("She greeted Aldous warmly. Aldous smiled.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        entry = next(e for e in lexicon if e.normalized_phrase == "aldous")
        assert entry.category == LexiconCategory.UNRESOLVED

    def test_possessive_surface_produces_base_form_phrase(self):
        # If the only surface form is a possessive ("Aldous's"), the lexicon
        # entry must use the base ("Aldous") as its phrase. The automaton
        # then matches "Aldous" in all contexts without creating a spurious
        # separate match inside "Aldous's" (the apostrophe boundary check
        # handles that).
        clusters = harvest_and_cluster("Aldous's sword was missing.")
        lexicon = induce_lexicon(clusters, "doc.md", induction_pass=0)
        entry = next(e for e in lexicon if e.normalized_phrase == "aldous")
        assert entry.phrase == "Aldous"

    def test_entry_id_is_stable_across_runs(self):
        # entry_id drives deduplication of lexicon records across pipeline runs.
        # Running induction twice on the same document must produce identical IDs.
        clusters = harvest_and_cluster("She greeted Aldous warmly. Aldous smiled.")
        lex1 = induce_lexicon(clusters, "doc.md", induction_pass=0)
        lex2 = induce_lexicon(clusters, "doc.md", induction_pass=0)
        ids1 = {e.entry_id for e in lex1}
        ids2 = {e.entry_id for e in lex2}
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# Matcher: phrase matching
# ---------------------------------------------------------------------------

class TestMatcher:
    def test_automaton_finds_phrase_in_span(self):
        # The basic contract: a phrase in the lexicon must be found when it
        # appears in span text.
        entry = make_entry("Aldous", "aldous")
        automaton = compile_automaton([entry])
        candidates = match_text(automaton, "She greeted Aldous warmly.", "doc.md", 0, 0)
        assert any(c.surface == "Aldous" for c in candidates)

    def test_match_offsets_point_into_raw_text(self):
        # Lexicon match anchors must use absolute document positions so that
        # doc.raw_text[anchor.start_char:anchor.end_char] == surface.
        # A span-relative offset would silently produce wrong retrieval results.
        span_text = "She greeted Aldous warmly."
        span_start = 10  # the span does not start at position 0 in the document
        entry = make_entry("Aldous", "aldous")
        automaton = compile_automaton([entry])
        candidates = match_text(automaton, span_text, "doc.md", 0, span_start)
        assert len(candidates) == 1
        c = candidates[0]
        full_text = " " * span_start + span_text
        assert full_text[c.anchor.start_char:c.anchor.end_char] == c.surface

    def test_word_boundary_rejects_prefix_match(self):
        # "Aldous" must not match inside "Aldousing" (a longer word that
        # happens to start with "Aldous"). Without this boundary check, the
        # automaton would produce spurious candidates wherever the name appears
        # as a substring of a longer word.
        entry = make_entry("Aldous", "aldous")
        automaton = compile_automaton([entry])
        candidates = match_text(automaton, "The Aldousing process began.", "doc.md", 0, 0)
        assert len(candidates) == 0

    def test_word_boundary_rejects_possessive_as_base_match(self):
        # When the lexicon has "Aldous" as a pattern, it must NOT match inside
        # "Aldous's" - the apostrophe-s would create a spurious candidate at
        # the same position as the harvester's possessive candidate, inflating
        # the occurrence count after deduplication fails (different end offsets).
        entry = make_entry("Aldous", "aldous")
        automaton = compile_automaton([entry])
        candidates = match_text(automaton, "Aldous's sword fell.", "doc.md", 0, 0)
        # "Aldous" should not match because "'" follows "Aldous" at the boundary.
        assert len(candidates) == 0

    def test_empty_automaton_returns_empty(self):
        # An empty lexicon must produce no candidates without crashing. An
        # exception here would halt the bootstrap loop on any document whose
        # first pass yields no inductable clusters.
        automaton = compile_automaton([])
        candidates = match_text(automaton, "Aldous arrived.", "doc.md", 0, 0)
        assert candidates == []

    def test_match_rule_source_is_lexicon(self):
        # Lexicon candidates must carry rule_source='lexicon' so that the
        # suppression logic in harvest_manuscript does not apply to them on
        # re-processing. If the rule_source were 'bare_capitalized', sentence-
        # initial lexicon matches would be silently suppressed in pass 1.
        entry = make_entry("Aldous", "aldous")
        automaton = compile_automaton([entry])
        candidates = match_text(automaton, "She greeted Aldous.", "doc.md", 0, 0)
        assert all(c.rule_source == 'lexicon' for c in candidates)

    def test_typographic_phrase_matches_normalised_text(self):
        # If a phrase contains typographic characters (e.g. "O’Brien" with
        # a curly apostrophe), it must match against the normalised span text
        # where the curly apostrophe becomes a straight one. Without normalising
        # the pattern, no match would ever be found in real manuscripts.
        entry = make_entry("O’Brien", "o'brien")  # curly apostrophe in phrase
        automaton = compile_automaton([entry])
        candidates = match_text(automaton, "She greeted O’Brien.", "doc.md", 0, 0)
        assert any("Brien" in c.surface for c in candidates)


# ---------------------------------------------------------------------------
# Bootstrap: convergence loop
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_bootstrap_produces_lexicon_for_evidenced_document(self):
        # A document with a clearly evidenced entity must produce at least one
        # lexicon entry. An empty lexicon after bootstrap means the loop failed
        # to induct anything, which would prevent second-pass improvement.
        doc = parse("doc.md", "She greeted Aldous warmly. Aldous smiled.")
        result = bootstrap(doc)
        assert len(result.lexicon) > 0

    def test_bootstrap_with_max_passes_one_runs_one_pass(self):
        # max_passes=1 disables the convergence loop (only pass 0 runs). This
        # is the primary mechanism for callers that want a single-pass harvest
        # without lexicon-guided re-matching.
        doc = parse("doc.md", "She greeted Aldous warmly. Aldous smiled.")
        result = bootstrap(doc, max_passes=1)
        assert result.passes_run == 1

    def test_new_entries_per_pass_length_equals_passes_run(self):
        # new_entries_per_pass records one count per pass. If its length
        # diverged from passes_run, any code that zip()s the two lists would
        # silently misalign pass metrics.
        doc = parse("doc.md", "She greeted Aldous warmly. Aldous smiled.")
        result = bootstrap(doc, max_passes=3)
        assert len(result.new_entries_per_pass) == result.passes_run

    def test_bootstrap_terminates_without_exceeding_max_passes(self):
        # The loop must stop at max_passes even if it has not converged.
        # A bug that ignored the cap could run indefinitely on a document
        # that always produces new entries.
        doc = parse("doc.md", "She greeted Aldous warmly. Aldous smiled.")
        result = bootstrap(doc, max_passes=2)
        assert result.passes_run <= 2

    def test_stopwords_absent_from_final_lexicon(self):
        # No stopword must appear in the final lexicon regardless of how many
        # times it appears in the document. A stopword in the lexicon would
        # generate false-positive candidates on every occurrence of that word.
        doc = parse("doc.md", "She arrived. She left. She returned. She waited.")
        result = bootstrap(doc)
        normalised_phrases = {e.normalized_phrase for e in result.lexicon}
        assert "she" not in normalised_phrases

    def test_bootstrap_result_clusters_match_lexicon_evidence(self):
        # The final clusters in BootstrapResult must be consistent with the
        # final lexicon: every lexicon normalized_phrase must correspond to a
        # cluster in the result.
        doc = parse("doc.md", "She greeted Aldous warmly. Aldous smiled.")
        result = bootstrap(doc)
        cluster_keys = {c.normalized_key for c in result.clusters}
        for entry in result.lexicon:
            assert entry.normalized_phrase in cluster_keys

    def test_bootstrap_pass_zero_new_entries_equals_initial_lexicon_size(self):
        # new_entries_per_pass[0] is the number of entries inducted in pass 0.
        # It must equal len(result.lexicon) when passes_run==1 (no new entries
        # can appear in later passes because they didn't run or converged to 0).
        doc = parse("doc.md", "She greeted Aldous warmly. Aldous smiled.")
        result = bootstrap(doc, max_passes=1)
        assert result.new_entries_per_pass[0] == len(result.lexicon)
