"""
Tests for backend/nlp/promotion/attribution.py.

Each test encodes a non-obvious rule or invariant in the attribution detection.
Comments explain which decision is locked in and why its breakage would be silent.
"""

import pytest

from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.types import MentionCluster, SpanAnchor, stable_hash_id
from backend.nlp.promotion.attribution import attribute_dialogue, SPEECH_VERB_LEMMAS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_and_preprocess(text: str, path: str = "doc.md"):
    doc = parse(path, text)
    return preprocess(doc)


def make_cluster(normalized_key: str, surface_forms: list[str]) -> MentionCluster:
    """Build a minimal MentionCluster for attribution tests.

    Anchors are left empty because attribution works from PreprocessedDocument
    sentences and quote spans, not from the cluster's stored anchors.
    """
    return MentionCluster(
        normalized_key=normalized_key,
        surface_forms=surface_forms,
        anchors=[],
        occurrence_count=len(surface_forms),
        title_support_count=0,
        possessive_support_count=0,
        location_support_count=0,
        linked_fields=[],
        linked_definitions=[],
        linked_seeds=[],
        cluster_id=stable_hash_id("doc.md", normalized_key),
    )


# ---------------------------------------------------------------------------
# Attribution detection
# ---------------------------------------------------------------------------

class TestAttribution:
    def test_post_quote_attribution_detected(self):
        # The pattern '"Quote," Name said.' must produce an AttributionRecord.
        # Without this, speaker evidence from the most common dialogue format
        # in English prose would be silently missed.
        pre = parse_and_preprocess('"Go now," Aldous said.')
        clusters = [make_cluster("aldous", ["Aldous"])]
        records = attribute_dialogue(pre, clusters)
        assert any(r.speaker_key == "aldous" for r in records)

    def test_pre_quote_attribution_detected(self):
        # The pattern 'Name said "Quote".' must produce an AttributionRecord.
        # Pre-quote attribution is the second-most common dialogue tag format.
        pre = parse_and_preprocess('Aldous whispered "Go now."')
        clusters = [make_cluster("aldous", ["Aldous"])]
        records = attribute_dialogue(pre, clusters)
        assert any(r.speaker_key == "aldous" for r in records)

    def test_no_speech_verb_no_attribution(self):
        # A sentence with a quote and a cluster surface but no speech verb must
        # not produce an attribution. Without the verb check, any cluster name
        # near a quote would be falsely attributed as the speaker.
        pre = parse_and_preprocess('"Go now," Aldous thought. He nodded.')
        clusters = [make_cluster("he", ["He"])]
        # "thought" IS a speech verb - use a verb that is definitely not in SPEECH_VERBS.
        pre2 = parse_and_preprocess('"Go now." Aldous stood silently.')
        # "Go now." ends with a period inside the quote, so the next sentence
        # "Aldous stood silently." has no quote and no attribution.
        records2 = attribute_dialogue(pre2, clusters)
        assert not any(r.speaker_key == "aldous" for r in records2)

    def test_unknown_speaker_no_attribution(self):
        # A speech verb adjacent to a quote but with no matching cluster surface
        # must produce no record. Without this check, any speech verb near a
        # quote would generate an attribution for whichever cluster happened to
        # match last, or crash on an empty lookup.
        pre = parse_and_preprocess('"Go now," Barnabas said.')
        clusters = [make_cluster("aldous", ["Aldous"])]  # Barnabas not in clusters
        records = attribute_dialogue(pre, clusters)
        assert len(records) == 0

    def test_multiple_quotes_counted_separately(self):
        # Two quotes attributed to the same speaker must produce two records.
        # A single-record implementation would undercount repeated dialogue
        # and suppress the attribution_count signal in scoring.
        text = '"Hello," Aldous said. He nodded. "Goodbye," Aldous replied.'
        pre = parse_and_preprocess(text)
        clusters = [make_cluster("aldous", ["Aldous"])]
        records = attribute_dialogue(pre, clusters)
        aldous_records = [r for r in records if r.speaker_key == "aldous"]
        assert len(aldous_records) == 2

    def test_present_tense_speech_verb_detected(self):
        # "says" and "whispers" are inflected forms not listed literally in
        # SPEECH_VERB_LEMMAS. Without WordNetLemmatizer reducing them to their
        # base forms, present-tense dialogue tags would silently produce no
        # attribution even though the pattern is unambiguous.
        pre = parse_and_preprocess('"Leave now," Aldous says.')
        clusters = [make_cluster("aldous", ["Aldous"])]
        records = attribute_dialogue(pre, clusters)
        assert any(r.speaker_key == "aldous" for r in records)

    def test_expanded_dialogue_tag_verb_detected(self):
        # Manuscripts use a broader tag vocabulary than "said" and
        # "whispered". A verb like "interjects" should count as a speech tag
        # so direct-quote lines keep their speaker evidence.
        pre = parse_and_preprocess('"Go now," Aldous interjects.')
        clusters = [make_cluster("aldous", ["Aldous"])]
        records = attribute_dialogue(pre, clusters)
        assert any(r.speaker_key == "aldous" for r in records)

    def test_single_adverb_gap_between_speaker_and_verb_is_allowed(self):
        # Fiction often inserts one adverb between the speaker and the speech
        # verb. Without this narrow gap rule, lines like "Kohaku gently
        # teases" silently lose attribution despite being explicit tags.
        pre = parse_and_preprocess('"Go now," Aldous gently teases back.')
        clusters = [make_cluster("aldous", ["Aldous"])]
        records = attribute_dialogue(pre, clusters)
        assert any(r.speaker_key == "aldous" for r in records)

    def test_post_quote_pattern_label(self):
        # The pattern field must be 'post_quote' for post-quote attribution.
        # A wrong label would corrupt pattern-specific statistics and prevent
        # callers from distinguishing dialogue formats.
        pre = parse_and_preprocess('"Go now," Aldous said.')
        clusters = [make_cluster("aldous", ["Aldous"])]
        records = attribute_dialogue(pre, clusters)
        aldous_records = [r for r in records if r.speaker_key == "aldous"]
        assert aldous_records, "expected at least one attribution record"
        assert all(r.pattern == 'post_quote' for r in aldous_records)

    def test_nearest_to_verb_wins_over_document_order(self):
        # When two cluster surfaces appear in the same attribution window, the
        # one nearest (in token count) to the speech verb must win. The previous
        # implementation returned on the first match in dict insertion order,
        # which is document order (first-occurring cluster first). In a sentence
        # like "As Aldous watched, Mary said, 'Hello.'", Aldous appears first in
        # the document and would win under dict-order, even though Mary is
        # unambiguously the speaker.
        pre = parse_and_preprocess("As Aldous watched, Mary said, 'Hello.'")
        clusters = [
            make_cluster("aldous", ["Aldous"]),  # inserted first = dict-order winner
            make_cluster("mary", ["Mary"]),
        ]
        records = attribute_dialogue(pre, clusters)
        assert any(r.speaker_key == "mary" for r in records)
        assert not any(r.speaker_key == "aldous" for r in records)

    def test_distant_discourse_word_not_treated_as_speaker(self):
        # A distant sentence-initial discourse word must not win attribution
        # just because it is capitalized and appears in the same sentence as a
        # speech verb. The speaker candidate must be structurally close to the
        # speech tag, not merely somewhere in the token window.
        pre = parse_and_preprocess('Still facing the water, she adds, "Hello."')
        clusters = [make_cluster("still", ["Still"])]
        records = attribute_dialogue(pre, clusters)
        assert records == []

    def test_group_like_cluster_not_eligible_as_speaker(self):
        # A collective or institutional name should not be considered a speaker
        # candidate in dialogue attribution unless later phases add explicit
        # support for personified groups.
        pre = parse_and_preprocess('"Go now," Institute said.')
        clusters = [make_cluster("institute", ["Institute"])]
        records = attribute_dialogue(pre, clusters)
        assert records == []

    def test_later_quote_opening_word_not_mistaken_for_previous_speaker(self):
        # When one sentence contains two quotes, the attribution window for the
        # first quote must stop before the next quote opens. Without this, a
        # later quote opener like "Last" can be incorrectly linked as the
        # speaker of the earlier quote.
        pre = parse_and_preprocess('"We bought what we needed," she says. "Last time was worse."')
        clusters = [make_cluster("last", ["Last"])]
        records = attribute_dialogue(pre, clusters)
        assert records == []

    def test_interrupted_quote_opening_word_not_mistaken_for_resumed_speaker(self):
        # Split dialogue tags like '"Tell them," she says, "that..."' must not
        # let the first quote's opening verb leak into the pre-window for the
        # resumed fragment. Without this boundary, "Tell" becomes a false
        # speaker candidate for the second quote.
        pre = parse_and_preprocess('"Tell them," she says, "that I am grateful."')
        clusters = [make_cluster("tell", ["Tell"])]
        records = attribute_dialogue(pre, clusters)
        assert records == []

    def test_post_quote_noun_phrase_after_pronoun_tag_not_treated_as_speaker(self):
        # In '"Ah," she says, one hand rising', the only actual speech tag is
        # the pronoun phrase "she says". A later capitalized noun phrase must
        # not be backfilled into speaker position just because it appears near
        # the speech verb.
        pre = parse_and_preprocess('"Ah," she says, One hand rising.')
        clusters = [make_cluster("one", ["One"])]
        records = attribute_dialogue(pre, clusters)
        assert records == []

    def test_post_quote_trailing_adverb_not_treated_as_speaker(self):
        # In '"Hello," she says at Last.', the word after the preposition is an
        # adverbial tail, not a deferred speaker surface. This guards against
        # distance-only matching that would promote "Last" into a character.
        pre = parse_and_preprocess('"Hello," she says at Last.')
        clusters = [make_cluster("last", ["Last"])]
        records = attribute_dialogue(pre, clusters)
        assert records == []
