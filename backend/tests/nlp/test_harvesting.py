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
from backend.nlp.harvesting.shared import (
    EVENT_NOUNS,
    GROUP_COLLECTIVE_VERBS,
    GROUP_LEADERSHIP_NOUNS,
    GROUP_MEMBERSHIP_VERBS,
    PLACE_DESCRIPTOR_NOUNS,
    PLACE_POSSESSIVE_CONTEXT_NOUNS,
    RELATION_ROLE_NOUNS,
    TITLE_PREFIXES,
    is_stopword,
    normalize_surface,
)


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


class TestPlaceDescriptorNouns:
    def test_wordnet_expansion_adds_generic_place_descriptors(self):
        # The place classifier should be able to recognize more than the small
        # hand-curated seed list. WordNet-backed expansion is meant to add
        # generic descriptor nouns such as "municipality" and "seaport"
        # without requiring a growing manual list.
        assert "municipality" in PLACE_DESCRIPTOR_NOUNS
        assert "seaport" in PLACE_DESCRIPTOR_NOUNS

    def test_wordnet_expansion_does_not_become_gazetteer(self):
        # PLACE_DESCRIPTOR_NOUNS is a common-noun descriptor list, not a
        # database of actual place names. Named entries like "amsterdam" would
        # make descriptor support behave like a noisy gazetteer.
        assert "amsterdam" not in PLACE_DESCRIPTOR_NOUNS
        assert "aachen" not in PLACE_DESCRIPTOR_NOUNS


class TestPlacePossessiveContextNouns:
    def test_wordnet_expansion_adds_place_feature_nouns(self):
        # Possessive place context should grow beyond the manual seed so the
        # place scorer can recognize common civic or terrain features.
        assert "harbour" in PLACE_POSSESSIVE_CONTEXT_NOUNS or "harbor" in PLACE_POSSESSIVE_CONTEXT_NOUNS
        assert "avenue" in PLACE_POSSESSIVE_CONTEXT_NOUNS

    def test_wordnet_expansion_does_not_become_owned_object_bag(self):
        # This set is for place-owned features, not arbitrary possessed nouns.
        # Words like "room" or "house" would make personal possession look
        # like place evidence too often.
        assert "room" not in PLACE_POSSESSIVE_CONTEXT_NOUNS
        assert "house" not in PLACE_POSSESSIVE_CONTEXT_NOUNS


class TestEventNouns:
    def test_wordnet_expansion_adds_generic_event_nouns(self):
        # EVENT_NOUNS should grow beyond the small manual seed so the event
        # scorer can recognize common event heads without hand-curating every
        # ceremony or occurrence term.
        assert "pageant" in EVENT_NOUNS
        assert "burial" in EVENT_NOUNS

    def test_wordnet_expansion_does_not_become_abstract_noun_bag(self):
        # EVENT_NOUNS should remain a concrete event-head list rather than
        # absorbing broad abstractions that would make event resolution noisy.
        assert "joy" not in EVENT_NOUNS
        assert "idea" not in EVENT_NOUNS
        assert "circumstance" not in EVENT_NOUNS


class TestGroupMembershipVerbs:
    def test_wordnet_expansion_adds_group_membership_verbs(self):
        # Group membership detection should grow beyond the manual seed so the
        # classifier can recognize collective affiliation prose without
        # hand-curating every verb form.
        assert "collaborate" in GROUP_MEMBERSHIP_VERBS
        assert "collaborated" in GROUP_MEMBERSHIP_VERBS

    def test_wordnet_expansion_does_not_become_generic_action_bag(self):
        # This set should stay focused on affiliation/membership language.
        # Drift into broad work verbs would make group resolution noisy.
        assert "whore" not in GROUP_MEMBERSHIP_VERBS
        assert "busy" not in GROUP_MEMBERSHIP_VERBS


class TestGroupLeadershipNouns:
    def test_wordnet_expansion_adds_collective_leadership_nouns(self):
        # Leadership framing should recognize common collective-role nouns
        # beyond the original hand-picked list.
        assert "manager" in GROUP_LEADERSHIP_NOUNS

    def test_wordnet_expansion_does_not_become_generic_person_role_bag(self):
        # This set should stay focused on leadership roles, not drift into
        # broad person-like nouns that would overfire on character prose.
        assert "hero" not in GROUP_LEADERSHIP_NOUNS


class TestGroupCollectiveVerbs:
    def test_wordnet_expansion_adds_collective_action_verbs(self):
        # Group-like action should recognize a few more institutional verbs
        # than the manual seed list alone.
        assert "regulate" in GROUP_COLLECTIVE_VERBS
        assert "regulated" in GROUP_COLLECTIVE_VERBS

    def test_wordnet_expansion_does_not_become_generic_motion_bag(self):
        # Collective-action detection should not absorb stray verb senses from
        # ambiguous roots like deploy or meet.
        assert "play" not in GROUP_COLLECTIVE_VERBS
        assert "see" not in GROUP_COLLECTIVE_VERBS


class TestRelationRoleNouns:
    def test_wordnet_expansion_adds_relation_role_nouns(self):
        # Relation-role references should cover more than the narrow seed list
        # so the semantic-review layer can preserve kinship language without
        # hand-curating every family variant.
        assert "auntie" in RELATION_ROLE_NOUNS
        assert "grandaunt" in RELATION_ROLE_NOUNS

    def test_wordnet_expansion_does_not_become_generic_social_noun_bag(self):
        # This set is for kinship and relation-role references, not broad
        # social nouns that would make bare-relation extraction noisy.
        assert "hero" not in RELATION_ROLE_NOUNS
        assert "friend" not in RELATION_ROLE_NOUNS


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
        # The bare "Aldous" appears mid-sentence so it survives the
        # all-sentence-initial suppression gate.
        candidates = pipeline("She saw Aldous. Aldous's sword fell.")
        keys = normalized_keys(candidates)
        assert "aldous" in keys
        assert len([c for c in candidates if c.normalized == "aldous"]) >= 2

    def test_terminal_s_possessive_produces_candidate(self):
        # "James'" tokenises as ["James", "'"] because the tokenizer regex
        # requires at least one \w after the apostrophe. Pass 2b detects this
        # two-token form via a lookahead: if an uppercase word ending in 's' is
        # immediately followed by an adjacent bare apostrophe, it is a possessive.
        # Without this, names like "James'", "Aldous'", and "soldiers'" would
        # silently produce no possessive candidate.
        candidates = pipeline("James' sword was missing.")
        poss = next((c for c in candidates if c.rule_source == 'possessive'), None)
        assert poss is not None
        assert poss.normalized == "james"
        assert poss.has_possessive is True


# ---------------------------------------------------------------------------
# Candidate extraction: contraction detection (possessive gate)
# ---------------------------------------------------------------------------

class TestContractionDetection:
    def test_contraction_not_harvested_as_possessive(self):
        # "Let's" is a contraction of "let us", not a possessive. Without the
        # contraction gate, Pass 2 would give it has_possessive=True and inflate
        # entityhood via spurious possessive support.
        candidates = pipeline('"Let\'s go," she said.')
        poss = [c for c in candidates if c.rule_source == 'possessive'
                and c.normalized == "let"]
        assert len(poss) == 0

    def test_real_possessive_still_harvested(self):
        # The contraction gate must not suppress genuine possessives.
        candidates = pipeline("Aldous's sword gleamed.")
        poss = [c for c in candidates if c.rule_source == 'possessive'
                and c.normalized == "aldous"]
        assert len(poss) == 1

    def test_contraction_base_bare_cap_still_harvested(self):
        # The contraction gate only blocks the possessive path. Pass 3 bare-cap
        # extraction must still harvest the token from non-possessive usage.
        candidates = pipeline("She saw Let. Let spoke quietly.")
        bare = [c for c in candidates if c.rule_source == 'bare_capitalized'
                and c.normalized == "let"]
        assert len(bare) >= 1


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

    def test_recurring_name_with_mid_sentence_occurrence_not_suppressed(self):
        # A name that appears at sentence-initial position AND mid-sentence
        # must not be suppressed. The mid-sentence occurrence is the evidence
        # that the capital reflects a proper name rather than sentence position.
        # A name appearing exclusively at sentence-initial position is suppressed
        # because the capitalisation is fully explained by position alone.
        candidates = pipeline("Aldous entered. She greeted Aldous warmly.")
        aldous_cands = [c for c in candidates if c.normalized == "aldous"]
        assert len(aldous_cands) >= 2

    def test_name_with_support_in_other_span_not_suppressed(self):
        # A name that appears sentence-initially in one span but mid-sentence
        # in another span must not be suppressed. The mid-sentence occurrence
        # provides the support needed to survive suppression.
        candidates = pipeline("Rhea sat.\n\nShe greeted Rhea warmly.")
        rhea_cands = [c for c in candidates if c.normalized == "rhea"]
        assert len(rhea_cands) >= 1

    def test_two_token_person_name_produces_compound_candidate(self):
        # Adjacent capitalized tokens should produce a multi-token compound
        # candidate in addition to their single-token candidates so later
        # stages can deterministically decide canonical identity.
        # The name appears mid-sentence to avoid sentence-initial suppression.
        candidates = pipeline("Then Tsushima Yoshiko arrived.")
        surfaces = {c.surface for c in candidates}
        normalized = {c.normalized for c in candidates}
        assert "Tsushima Yoshiko" in surfaces
        assert "tsushima yoshiko" in normalized
        assert "tsushima" in normalized
        assert "yoshiko" in normalized

    def test_two_token_group_name_produces_compound_candidate(self):
        # Institutional compounds such as "Norre Institute" should be
        # harvested as coherent surfaces, not only as isolated tokens.
        candidates = pipeline("The Norre Institute reopened.")
        assert any(
            c.surface == "Norre Institute" and c.rule_source == "compound_capitalized"
            for c in candidates
        )

    def test_three_token_name_produces_compound_candidate(self):
        # Longer contiguous capitalized phrases should survive as one compound
        # candidate so later overlap resolution can prefer the full surface
        # over generic fragments like "Old Man" or "Man Hiroshi".
        # The name appears mid-sentence to avoid sentence-initial suppression.
        candidates = pipeline("The shop of Old Man Hiroshi smelled of dust.")
        assert any(
            c.normalized == "old man hiroshi" and c.rule_source == "compound_capitalized"
            for c in candidates
        )

    def test_compound_candidates_do_not_cross_punctuation_boundaries(self):
        # Compound harvesting must stay within contiguous token spans.
        # Punctuation-separated title-case tokens are too ambiguous to join.
        candidates = pipeline("Tsushima, Yoshiko arrived.")
        assert not any(c.surface == "Tsushima Yoshiko" for c in candidates)

    def test_sentence_initial_only_compound_suppressed(self):
        # A compound like "Even Kohaku" that only ever appears at the start of
        # a sentence provides no evidence of a proper name - the capitalisation
        # is explained entirely by sentence position. Without this suppression,
        # sentence-opening adverb + name pairs ("Still Kohaku", "Even Kohaku")
        # would survive as phantom compound entities.
        candidates = pipeline(
            "Even Kohaku smiled. Even Kohaku waved."
        )
        assert not any(
            c.rule_source == "compound_capitalized"
            and c.normalized == "even kohaku"
            for c in candidates
        )

    def test_compound_with_mid_sentence_occurrence_preserved(self):
        # A compound that appears at least once mid-sentence has positional
        # evidence that the capitalisation reflects a proper name. It must
        # survive suppression regardless of how many sentence-initial
        # occurrences also exist.
        candidates = pipeline(
            "Tsushima Yoshiko arrived. She greeted Tsushima Yoshiko warmly."
        )
        compound_cands = [
            c for c in candidates
            if c.rule_source == "compound_capitalized"
            and c.normalized == "tsushima yoshiko"
        ]
        assert len(compound_cands) >= 1

    def test_bare_cap_not_protected_by_sentence_initial_only_compound(self):
        # "Even" only appears sentence-initially and its sole compound
        # participation is "Even Kohaku", which is also sentence-initial-only.
        # A suppressed compound must not provide structural support for its
        # component bare-cap tokens. Without this rule, sentence-opening
        # adverbs would survive suppression by riding the compound they
        # accidentally form.
        candidates = pipeline(
            "Even Kohaku smiled. Even Kohaku paused."
        )
        assert not any(
            c.rule_source == "bare_capitalized"
            and c.normalized == "even"
            for c in candidates
        )

    def test_bare_cap_protected_by_surviving_compound(self):
        # "Tsushima" only appears sentence-initially as a bare token, but
        # participates in "Tsushima Yoshiko" which has a mid-sentence
        # occurrence. The surviving compound provides structural support, so
        # "Tsushima" must not be suppressed. Without this, component tokens
        # of legitimate compound names would vanish whenever they lack
        # independent mid-sentence evidence.
        candidates = pipeline(
            "Tsushima Yoshiko arrived. She greeted Tsushima Yoshiko warmly. "
            "Tsushima paused."
        )
        bare_tsushima = [
            c for c in candidates
            if c.rule_source == "bare_capitalized"
            and c.normalized == "tsushima"
        ]
        assert len(bare_tsushima) >= 1


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
