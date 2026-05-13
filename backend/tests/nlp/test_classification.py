"""
Tests for backend/nlp/classification/*.

These tests lock in the top-level classification contract and the arbitration
rules that separate character/group/place decisions from the rest of the NLP
pipeline.
"""

from backend.nlp.classification.arbitration import (
    classify_cluster,
    classify_clusters,
)
from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.harvesting.manuscript import harvest_manuscript
from backend.nlp.clustering.clustering import cluster_mentions
from backend.nlp.clustering.linking import link_clusters
from backend.nlp.promotion.attribution import attribute_dialogue
from backend.nlp.types import (
    DefinitionCandidate,
    LexiconCategory,
    MentionCluster,
    SpanAnchor,
    stable_hash_id,
)


def harvest_and_cluster(text: str, path: str = "doc.md"):
    doc = parse(path, text)
    pre = preprocess(doc)
    candidates = harvest_manuscript(pre)
    clusters = cluster_mentions(candidates)
    link_clusters(clusters, [], [], [])
    return pre, clusters


def harvest_and_cluster_with_definitions(
    text: str,
    definitions: list[DefinitionCandidate],
    path: str = "doc.md",
):
    """Run harvest/cluster/link with explicit definition candidates."""
    doc = parse(path, text)
    pre = preprocess(doc)
    candidates = harvest_manuscript(pre)
    clusters = cluster_mentions(candidates)
    link_clusters(clusters, [], definitions, [])
    return pre, clusters


def make_cluster(
    normalized_key: str,
    *,
    occurrence_count: int = 1,
    has_title_support: bool = False,
    has_possessive_support: bool = False,
    has_location_support: bool = False,
) -> MentionCluster:
    """Build a minimal cluster for classification-only tests."""
    return MentionCluster(
        normalized_key=normalized_key,
        surface_forms=[normalized_key.title()],
        anchors=[],
        occurrence_count=occurrence_count,
        has_title_support=has_title_support,
        has_possessive_support=has_possessive_support,
        has_location_support=has_location_support,
        linked_fields=[],
        linked_definitions=[],
        linked_seeds=[],
        cluster_id=stable_hash_id("doc.md", normalized_key),
    )


class TestClassification:
    def test_titled_cluster_resolves_character(self):
        # A titled entity is the clearest deterministic person signal.
        pre, clusters = harvest_and_cluster("She met Captain Aldous by the gate.")
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        decision = classify_cluster(aldous, pre, [])
        assert decision.winning_category == LexiconCategory.CHARACTER
        assert decision.resolved is True

    def test_locative_cluster_resolves_place_without_attribution(self):
        # A capitalized name that appears after a locative preposition should
        # resolve as PLACE when no stronger person evidence exists.
        pre, clusters = harvest_and_cluster("She arrived in Tairngire. Tairngire glowed.")
        tairngire = next(c for c in clusters if c.normalized_key == "tairngire")
        decision = classify_cluster(tairngire, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_attribution_beats_place_evidence(self):
        # If a cluster is attributed as a speaker, personhood outranks locative
        # context from some other occurrence.
        text = 'She stood in Aldous Hall. "Go now," Aldous said.'
        pre, clusters = harvest_and_cluster(text)
        records = attribute_dialogue(pre, clusters)
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        decision = classify_cluster(aldous, pre, records)
        assert decision.winning_category == LexiconCategory.CHARACTER
        assert decision.resolved is True

    def test_group_suffix_resolves_group(self):
        # Institutional suffixes should resolve to the collective group class,
        # not to the legacy faction-specific category.
        pre, clusters = harvest_and_cluster(
            "The Norre Institute closed its gates. The Institute remained silent."
        )
        institute = next(c for c in clusters if c.normalized_key == "institute")
        decision = classify_cluster(institute, pre, [])
        assert decision.winning_category == LexiconCategory.GROUP
        assert decision.resolved is True

    def test_membership_and_collective_action_resolve_group(self):
        # A non-suffix name should still resolve as GROUP when the prose gives
        # membership context and collective action verbs.
        text = "She served with Meridian. Meridian deployed scouts at dawn."
        pre, clusters = harvest_and_cluster(text)
        meridian = next(c for c in clusters if c.normalized_key == "meridian")
        decision = classify_cluster(meridian, pre, [])
        assert decision.winning_category == LexiconCategory.GROUP
        assert decision.resolved is True

    def test_leadership_phrase_resolves_group(self):
        # Leadership framing like "leader of Meridian" should support group
        # resolution even when the name itself has no faction-like suffix.
        text = "The leader of Meridian arrived. Meridian governed the coast."
        pre, clusters = harvest_and_cluster(text)
        meridian = next(c for c in clusters if c.normalized_key == "meridian")
        decision = classify_cluster(meridian, pre, [])
        assert decision.winning_category == LexiconCategory.GROUP
        assert decision.resolved is True

    def test_compound_person_name_without_behavior_stays_unresolved(self):
        # Structural shape and recurrence alone are no longer enough to resolve
        # character identity. This keeps non-character compounds from being
        # overclassified as CHARACTER when no behavioral evidence is present.
        text = "Tsushima Yoshiko arrived. Tsushima Yoshiko nodded."
        pre, clusters = harvest_and_cluster(text)
        yoshiko = next(c for c in clusters if c.normalized_key == "tsushima yoshiko")
        decision = classify_cluster(yoshiko, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False

    def test_compound_group_name_resolves_group(self):
        # A compound institutional name should carry its suffix semantics as a
        # whole surface, not only through the head noun token in isolation.
        text = "The Norre Institute reopened. Norre Institute deployed scouts."
        pre, clusters = harvest_and_cluster(text)
        institute = next(c for c in clusters if c.normalized_key == "norre institute")
        decision = classify_cluster(institute, pre, [])
        assert decision.winning_category == LexiconCategory.GROUP
        assert decision.resolved is True

    def test_compound_event_name_resolves_event(self):
        # Event compounds should resolve from the full surface when temporal
        # framing and occurrence verbs support them.
        text = (
            "The Lantern Festival began at dusk. During the Lantern Festival, bells rang."
        )
        pre, clusters = harvest_and_cluster(text)
        festival = next(c for c in clusters if c.normalized_key == "lantern festival")
        decision = classify_cluster(festival, pre, [])
        assert decision.winning_category == LexiconCategory.EVENT
        assert decision.resolved is True

    def test_place_descriptor_compound_resolves_place(self):
        # A compound whose head is a geographic descriptor should resolve as a
        # place even when the descriptor is internal to the entity span rather
        # than appearing in surrounding prose.
        text = "East Lagoon shimmered at dawn. East Lagoon slept by noon."
        pre, clusters = harvest_and_cluster(text)
        lagoon = next(c for c in clusters if c.normalized_key == "east lagoon")
        decision = classify_cluster(lagoon, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_directional_compound_resolves_place(self):
        # Directional compounds like "Polar North" are place-like as complete
        # names even when they do not appear after locative prepositions.
        text = "Polar North glittered at dawn. The Polar North darkened again."
        pre, clusters = harvest_and_cluster(text)
        north = next(c for c in clusters if c.normalized_key == "polar north")
        decision = classify_cluster(north, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_compound_event_head_resolves_event(self):
        # Named event compounds should gain event evidence from their head noun
        # rather than waiting for separate temporal framing every time.
        text = "Lantern Festival returned at dusk. The Lantern Festival filled the harbor."
        pre, clusters = harvest_and_cluster(text)
        festival = next(c for c in clusters if c.normalized_key == "lantern festival")
        decision = classify_cluster(festival, pre, [])
        assert decision.winning_category == LexiconCategory.EVENT
        assert decision.resolved is True
        assert decision.entityhood.accepted is True

    def test_wordnet_expanded_group_terms_resolve_group(self):
        # NLTK-backed group lexicons should drive real classification behavior,
        # not just expand constants on paper. "collaborated with" and
        # "regulated" are broader collective cues than the original manual
        # seed set.
        text = "She collaborated with Meridian. Meridian regulated the coast."
        pre, clusters = harvest_and_cluster(text)
        meridian = next(c for c in clusters if c.normalized_key == "meridian")
        decision = classify_cluster(meridian, pre, [])
        assert decision.winning_category == LexiconCategory.GROUP
        assert decision.resolved is True

    def test_place_descriptor_support_resolves_place(self):
        # A geographic descriptor around a capitalized name should be enough
        # to resolve placehood even without a preceding locative preposition.
        pre, clusters = harvest_and_cluster("The city of Sidhe slept beneath the fog.")
        sidhe = next(c for c in clusters if c.normalized_key == "sidhe")
        decision = classify_cluster(sidhe, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_possessive_place_context_resolves_place(self):
        # Possessive context such as "Numazu's streets" should count as place
        # evidence rather than being left unresolved.
        text = "Numazu's streets glowed. She missed Numazu."
        pre, clusters = harvest_and_cluster(text)
        numazu = next(c for c in clusters if c.normalized_key == "numazu")
        decision = classify_cluster(numazu, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_heart_of_pattern_resolves_place(self):
        # Phrases like "heart of Numazu" are place framing even when the name
        # is not directly preceded by a standard location descriptor noun.
        text = "The heart of Numazu was loud. Numazu shimmered at dusk."
        pre, clusters = harvest_and_cluster(text)
        numazu = next(c for c in clusters if c.normalized_key == "numazu")
        decision = classify_cluster(numazu, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_civic_resident_pattern_resolves_place(self):
        # Resident nouns like "citizens" should support place resolution for
        # names such as "Numazu citizens".
        text = "Numazu citizens gathered at dawn. She left Numazu by noon."
        pre, clusters = harvest_and_cluster(text)
        numazu = next(c for c in clusters if c.normalized_key == "numazu")
        decision = classify_cluster(numazu, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_temporal_event_noun_resolves_event(self):
        # A recurring capitalized event noun with temporal framing and
        # occurrence language should resolve as EVENT rather than remaining
        # generic unresolved review noise.
        text = (
            "The Festival began at dusk. During the Festival, bells rang."
        )
        pre, clusters = harvest_and_cluster(text)
        festival = next(c for c in clusters if c.normalized_key == "festival")
        decision = classify_cluster(festival, pre, [])
        assert decision.winning_category == LexiconCategory.EVENT
        assert decision.resolved is True

    def test_event_noun_without_temporal_context_stays_unresolved(self):
        # Event-like nouns should not resolve as EVENT from capitalization
        # alone. Without temporal or occurrence framing they are too broad to
        # classify safely.
        text = (
            "They admired Festival lanterns. The boat carried Festival "
            "ribbons against the railing."
        )
        pre, clusters = harvest_and_cluster(text)
        festival = next(c for c in clusters if c.normalized_key == "festival")
        decision = classify_cluster(festival, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False

    def test_definition_style_term_resolves_concept(self):
        # A capitalized term followed by definition syntax and an abstract
        # descriptor should resolve as CONCEPT rather than staying unresolved.
        text = (
            "The term Leva refers to a magical resonance system. "
            "Leva destabilized the chamber."
        )
        pre, clusters = harvest_and_cluster(text)
        leva = next(c for c in clusters if c.normalized_key == "leva")
        decision = classify_cluster(leva, pre, [])
        assert decision.winning_category == LexiconCategory.CONCEPT
        assert decision.resolved is True

    def test_compound_concept_head_resolves_concept(self):
        # A compound with an explicitly abstract head noun such as "Protocol"
        # should resolve as a concept even without separate glossary syntax.
        text = "Leva Protocol failed at dusk. The Leva Protocol destabilized the chamber."
        pre, clusters = harvest_and_cluster(text)
        protocol = next(c for c in clusters if c.normalized_key == "leva protocol")
        decision = classify_cluster(protocol, pre, [])
        assert decision.winning_category == LexiconCategory.CONCEPT
        assert decision.resolved is True

    def test_linked_definition_resolves_concept(self):
        # Structured definition notes are strong concept evidence even when
        # prose context is sparse.
        definition = DefinitionCandidate(
            term="Azoth",
            definition_text="A particulate energy protocol used in solunar rites.",
            anchor=SpanAnchor(path="doc.md", span_ordinal=0, start_char=0, end_char=5),
            candidate_id=stable_hash_id("doc.md", "0", "Azoth"),
        )
        text = "Azoth shimmered in the air. Later, Azoth failed."
        pre, clusters = harvest_and_cluster_with_definitions(text, [definition])
        azoth = next(c for c in clusters if c.normalized_key == "azoth")
        decision = classify_cluster(azoth, pre, [])
        assert decision.winning_category == LexiconCategory.CONCEPT
        assert decision.resolved is True

    def test_abstractish_word_without_definition_context_stays_unresolved(self):
        # A capitalized abstract-looking word should not resolve as CONCEPT
        # from recurrence alone. It still needs definition-style context.
        text = "Guide lights flickered. She picked up the Guide."
        pre, clusters = harvest_and_cluster(text)
        guide = next(c for c in clusters if c.normalized_key == "guide")
        decision = classify_cluster(guide, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False

    def test_possessive_only_cluster_remains_unresolved(self):
        # Possessive form alone is evidence of entityhood but not enough to
        # choose between person, place, object, or concept.
        pre, clusters = harvest_and_cluster("Aldous's sword was missing.")
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        decision = classify_cluster(aldous, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False
        assert decision.entityhood.accepted is True

    def test_possessive_owned_object_does_not_resolve_place(self):
        # Possessive nouns should not become places just because they precede
        # a common noun. Owned-object syntax such as "Aldous's room" is not
        # the same as "Numazu's streets".
        text = "Aldous's room was dark. She found Aldous later."
        pre, clusters = harvest_and_cluster(text)
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        decision = classify_cluster(aldous, pre, [])
        assert decision.winning_category != LexiconCategory.PLACE

    def test_collective_verb_alone_does_not_force_group(self):
        # A single collective-ish verb without membership or leadership
        # framing should not be enough to force a non-suffix name into GROUP.
        text = "Meridian shimmered at dusk. Later, Meridian moved again."
        pre, clusters = harvest_and_cluster(text)
        meridian = next(c for c in clusters if c.normalized_key == "meridian")
        decision = classify_cluster(meridian, pre, [])
        assert decision.winning_category != LexiconCategory.GROUP

    def test_recurring_bare_cluster_is_weak_entityhood(self):
        # Recurrence alone is not enough to call a cluster a trustworthy entity.
        # Weak recurring bare-cap clusters should remain visible to diagnostics
        # as unresolved, but they must not be treated as accepted entityhood.
        cluster = make_cluster("still", occurrence_count=4)
        decision = classify_cluster(cluster, None, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False
        assert decision.entityhood.accepted is False

    def test_weak_locative_compound_does_not_resolve_place(self):
        # A capitalized adjective inside an abstract compound such as
        # "Cosmic Time" must not resolve as a place just because it follows a
        # weak path preposition.
        pre, clusters = harvest_and_cluster("She fell through Cosmic Time.")
        cosmic = next(c for c in clusters if c.normalized_key == "cosmic")
        decision = classify_cluster(cosmic, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False
        assert decision.entityhood.accepted is False

    def test_demonym_like_cluster_after_weak_locative_stays_unresolved(self):
        # Demonym or adjectival forms such as "Lunarian" should not resolve as
        # places from a weak preposition alone.
        text = (
            "Tea was gathered by Lunarian druids. "
            "Later, Lunarian chants faded into the valley."
        )
        pre, clusters = harvest_and_cluster(text)
        lunarian = next(c for c in clusters if c.normalized_key == "lunarian")
        decision = classify_cluster(lunarian, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False
        assert decision.entityhood.accepted is False

    def test_bulk_classification_returns_mapping_by_cluster_key(self):
        pre, clusters = harvest_and_cluster("She met Captain Aldous in Tairngire.")
        decisions = classify_clusters(clusters, pre, [])
        assert set(decisions) == {c.normalized_key for c in clusters}
