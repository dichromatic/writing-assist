# Diagram omitted - utility module with no significant information flow.

"""
Shared constants and utilities for all harvesting modules.

All title prefix lists, field labels, stopwords, and suppression helpers live
here. Harvesting modules import from this file only - never define their own
copies of these constants.
"""

from __future__ import annotations

from functools import lru_cache

from backend.nlp.types import stable_hash_id  # noqa: F401 - re-exported for harvesters

# ---------------------------------------------------------------------------
# Title prefix list
#
# Used to identify name-like tokens in prose ("Captain Aldous") and to strip
# the prefix when computing the normalized clustering key ("aldous").
# ---------------------------------------------------------------------------
TITLE_PREFIXES: frozenset[str] = frozenset({
    'Mr', 'Mrs', 'Ms', 'Miss',
    'Dr', 'Prof', 'Professor',
    'Rev', 'Reverend', 'Father',
    'Captain', 'Capt', 'Cpt',
    'Lieutenant', 'Lt',
    'Sergeant', 'Sgt',
    'Colonel', 'Col',
    'General', 'Gen',
    'Admiral', 'Adm',
    'Lord', 'Lady',
    'Sir', 'Dame',
    'King', 'Queen', 'Prince', 'Princess',
    'Duke', 'Duchess', 'Baron', 'Baroness',
    'Count', 'Countess',
    'Saint', 'St',
})

# Reused lowercase projection of title prefixes. Centralizing this avoids
# duplicate set construction across classification, promotion, and review
# modules that all apply the same membership test.
TITLE_PREFIXES_LOWER: frozenset[str] = frozenset(
    title.lower() for title in TITLE_PREFIXES
)

# ---------------------------------------------------------------------------
# Relation-role nouns
#
# These nouns behave like deferred character references rather than stable
# canonicals in fiction prose. They are extracted for later semantic review so
# that kinship and role language such as "mother" or "mentor" is preserved
# even when it does not appear as a proper name.
# ---------------------------------------------------------------------------
_BASE_RELATION_ROLE_NOUNS: frozenset[str] = frozenset({
    "father", "mother", "parent", "sibling", "child", "spouse",
    "brother", "sister", "son", "daughter", "aunt", "uncle", "cousin",
    "mentor", "benefactor", "patron", "guardian", "ward", "master",
    "apprentice",
})

# ---------------------------------------------------------------------------
# Locative prepositions
#
# Used during harvesting to detect when a bare-capitalized token appears in a
# position that strongly indicates a place name. The preceding token is checked
# against this set in Pass 3 of the manuscript harvester.
#
# "to", "toward", "towards", and "for" are deliberately excluded: they precede
# people as often as places ("she spoke to Aldous", "he ran toward Kohaku")
# and would produce too many false positives.
# ---------------------------------------------------------------------------
LOCATIVE_PREPOSITIONS: frozenset[str] = frozenset({
    # Position
    "in", "at", "on", "near", "by", "beside", "between", "among",
    "amid", "amidst", "within", "outside", "inside", "around",
    "across", "along", "through", "beyond", "beneath", "below",
    "above", "over", "under", "underneath", "opposite", "behind",
    # Origin and path - reliable place indicators even when directional
    "from", "into", "onto", "throughout",
})

# ---------------------------------------------------------------------------
# Place-context refinements
#
# The manuscript harvester records only a coarse "has location context" flag.
# Later classifiers refine that signal using the actual neighboring tokens.
# Strong locatives can resolve a place on their own; weak locatives need
# corroboration because they also introduce abstract compounds ("through
# Cosmic Time") and adjectival demonyms ("by Lunarian druids").
# ---------------------------------------------------------------------------
STRONG_LOCATIVE_PREPOSITIONS: frozenset[str] = frozenset({
    "in", "at", "from", "into", "onto", "within", "inside", "outside",
    "near", "on",
})

WEAK_LOCATIVE_PREPOSITIONS: frozenset[str] = LOCATIVE_PREPOSITIONS - STRONG_LOCATIVE_PREPOSITIONS

_BASE_PLACE_DESCRIPTOR_NOUNS: frozenset[str] = frozenset({
    "city", "capital", "planet", "continent", "forest", "valley", "mountain",
    "range", "lagoon", "bay", "beach", "beaches", "street", "streets",
    "garden", "gardens", "village", "town", "port", "groundport",
})

_BASE_PLACE_POSSESSIVE_CONTEXT_NOUNS: frozenset[str] = frozenset({
    "street", "streets", "road", "roads", "shore", "shores", "sand", "sands",
    "harbor", "harbour", "bay", "beach", "beaches", "forest", "forests",
    "garden", "gardens", "capital", "port",
})

PLACE_RESIDENT_NOUNS: frozenset[str] = frozenset({
    "citizen", "citizens", "resident", "residents", "local", "locals",
    "inhabitant", "inhabitants",
})

DEMONYM_SUFFIXES: frozenset[str] = frozenset({
    "ian", "an", "ish", "ese",
})

# ---------------------------------------------------------------------------
# Event-context refinements
#
# Event classification needs narrow lexical support because capitalized event
# nouns such as "Festival" or "Remembrance" otherwise look like generic bare
# entities. These sets intentionally encode only stable local cues: event-like
# head nouns, temporal framing words, and nearby occurrence verbs.
# ---------------------------------------------------------------------------
_BASE_EVENT_NOUNS: frozenset[str] = frozenset({
    "festival", "ceremony", "remembrance", "memorial", "battle", "war",
    "mission", "ritual", "celebration", "coronation", "pilgrimage",
    "funeral", "trial", "summit", "expedition",
})

EVENT_TEMPORAL_PREPOSITIONS: frozenset[str] = frozenset({
    "during", "before", "after", "since", "until",
})

EVENT_INSTANCE_MARKERS: frozenset[str] = frozenset({
    "annual", "yearly", "nightly", "daily", "weekly", "monthly",
    "last", "next", "first", "final", "opening", "closing",
})

EVENT_OCCURRENCE_VERBS: frozenset[str] = frozenset({
    "begin", "begins", "began", "start", "starts", "started",
    "end", "ends", "ended",
    "held", "hold", "holds",
    "celebrate", "celebrates", "celebrated",
    "observe", "observes", "observed",
    "mark", "marks", "marked",
    "attend", "attends", "attended",
    "resume", "resumes", "resumed",
})

# ---------------------------------------------------------------------------
# Concept-context refinements
#
# Concept classification targets named abstract systems, rules, energies, and
# glossary-like terms. The local rules depend on definition verbs and abstract
# descriptor nouns rather than on generic recurrence.
# ---------------------------------------------------------------------------
CONCEPT_DEFINITION_VERBS: frozenset[str] = frozenset({
    "is", "was", "means", "meant", "refers", "describe", "describes",
    "described", "denotes", "denoted",
})

CONCEPT_DESCRIPTOR_NOUNS: frozenset[str] = frozenset({
    "system", "theory", "protocol", "resonance", "energy", "force",
    "discipline", "practice", "condition", "technique", "method",
    "concept", "term", "law", "principle", "process",
})

# ---------------------------------------------------------------------------
# Group-context refinements
#
# Group classification needs contextual cues for collective entities whose
# names do not end in a faction-like suffix. The lexical inventories use the
# same conservative WordNet-backed expansion pattern as place and event sets,
# while the prepositions remain manual because they are syntactic markers.
# ---------------------------------------------------------------------------
_BASE_GROUP_MEMBERSHIP_VERBS: frozenset[str] = frozenset({
    "serve", "served", "serves", "join", "joined", "joins", "work", "worked",
    "works", "fight", "fought", "fights",
})

GROUP_MEMBERSHIP_PREPOSITIONS: frozenset[str] = frozenset({
    "with", "under", "for",
})

_BASE_GROUP_LEADERSHIP_NOUNS: frozenset[str] = frozenset({
    "leader", "head", "chief", "captain", "director", "commander",
})

_BASE_GROUP_COLLECTIVE_VERBS: frozenset[str] = frozenset({
    "govern", "governed", "governs", "deploy", "deployed", "deploys",
    "meet", "met", "meets", "rule", "ruled", "rules", "found", "founded",
    "founds", "command", "commanded", "commands", "forbid", "forbade",
    "forbids",
})

# ---------------------------------------------------------------------------
# Dialogue attribution speech verbs
#
# Quote attribution needs a wider set than the original dialogue-tag list to
# catch common fiction verbs such as "interjects", "deadpans", and "teases".
# This still stays conservative because it runs inside small quote windows and
# should not drift into general communication verbs.
# ---------------------------------------------------------------------------
_BASE_SPEECH_VERB_LEMMAS: frozenset[str] = frozenset({
    "say", "ask", "reply", "answer", "call", "shout",
    "whisper", "cry", "murmur", "declare", "announce",
    "tell", "explain", "continue", "add", "admit", "agree",
    "warn", "suggest", "insist", "demand", "plead",
    "argue", "laugh", "sigh", "mutter", "think",
    "respond", "exclaim", "state", "note",
    "interject", "deadpan", "tease", "observe", "muse",
    "remark", "counter", "begin",
})


def _regular_verb_inflections(lemma: str) -> set[str]:
    """Generate simple present/past inflections for a base verb lemma."""
    forms = {lemma}

    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        forms.add(f"{lemma[:-1]}ies")
        forms.add(f"{lemma[:-1]}ied")
    elif lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        forms.add(f"{lemma}es")
        forms.add(f"{lemma}ed")
    elif lemma.endswith("e"):
        forms.add(f"{lemma}s")
        forms.add(f"{lemma}d")
    else:
        forms.add(f"{lemma}s")
        forms.add(f"{lemma}ed")

    return forms


_IRREGULAR_VERB_FORMS: dict[str, frozenset[str]] = {
    "fight": frozenset({"fight", "fights", "fought"}),
    "meet": frozenset({"meet", "meets", "met"}),
    "forbid": frozenset({"forbid", "forbids", "forbade"}),
    "lead": frozenset({"lead", "leads", "led"}),
}


def _expand_group_verbs(
    seed_synset_names: tuple[str, ...],
    fallback: frozenset[str],
    blacklist: set[str],
) -> frozenset[str]:
    """Build a conservative verb inventory for group-context detection."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return fallback

    try:
        verbs = set(fallback)
        for synset_name in seed_synset_names:
            synset = wn.synset(synset_name)
            candidate_synsets = [synset] + synset.hyponyms()
            for candidate in candidate_synsets:
                for lemma in candidate.lemmas():
                    word = lemma.name().lower()
                    if "_" in word or not word.isalpha():
                        continue
                    if len(word) < 3 or word in blacklist:
                        continue
                    verbs.update(_IRREGULAR_VERB_FORMS.get(word, _regular_verb_inflections(word)))
        return frozenset(verbs)
    except Exception:
        return fallback


def _expand_seeded_verbs(
    seed_synset_names: tuple[str, ...],
    fallback: frozenset[str],
    blacklist: set[str],
) -> frozenset[str]:
    """Build a conservative lemma inventory from seed synsets plus hyponyms."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return fallback

    try:
        verbs = set(fallback)
        for synset_name in seed_synset_names:
            synset = wn.synset(synset_name)
            candidate_synsets = [synset] + synset.hyponyms()
            for candidate in candidate_synsets:
                for lemma in candidate.lemmas():
                    word = lemma.name().lower()
                    if "_" in word or not word.isalpha():
                        continue
                    if len(word) < 3 or word in blacklist:
                        continue
                    verbs.add(word)
        return frozenset(verbs)
    except Exception:
        return fallback


def _load_group_membership_verbs() -> frozenset[str]:
    """Build a narrow affiliation/membership verb set for group detection."""
    blacklist = {
        "busy", "whore", "wait", "waits", "waited", "waitress", "minister",
        "page", "pages", "paged", "occupy", "occupies", "occupied",
        "carpenter", "carpenters", "carpentered", "clerk", "clerks",
        "clerked", "monkey", "monkeys", "monkeyed", "putter", "putters",
        "puttered", "potter", "potters", "pottered", "beaver", "beavers",
        "beavered", "boondoggle", "boondoggles", "boondoggled", "intern",
        "interns", "interned",
    }
    return _expand_group_verbs(
        ("join.v.01", "work.v.01", "fight.v.01"),
        _BASE_GROUP_MEMBERSHIP_VERBS,
        blacklist,
    )


def _load_group_leadership_nouns() -> frozenset[str]:
    """Build a narrow leadership-role noun set for group framing."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return _BASE_GROUP_LEADERSHIP_NOUNS

    blacklist = {
        "hero", "superman", "model", "father", "guru", "guide", "politician",
        "politico", "trainer", "caller", "cheerleader", "misleader",
        "torchbearer", "lawgiver", "lawmaker", "imam", "imaum", "demigod",
        "aristocrat", "patrician", "superior", "superordinate", "employer",
    }

    try:
        nouns = set(_BASE_GROUP_LEADERSHIP_NOUNS)
        for synset_name in ("leader.n.01", "director.n.01", "commander.n.01"):
            synset = wn.synset(synset_name)
            candidate_synsets = [synset] + synset.hyponyms()
            for candidate in candidate_synsets:
                for lemma in candidate.lemmas():
                    word = lemma.name().lower()
                    if "_" in word or not word.isalpha():
                        continue
                    if len(word) < 4 or word in blacklist:
                        continue
                    nouns.add(word)
        return frozenset(nouns)
    except Exception:
        return _BASE_GROUP_LEADERSHIP_NOUNS


def _load_group_collective_verbs() -> frozenset[str]:
    """Build a narrow institutional action verb set for group behavior."""
    blacklist = {
        "play", "plays", "played", "see", "sees", "saw", "cross", "crosses",
        "crossed", "intersect", "intersects", "intersected", "district",
        "districts", "districted", "order", "orders", "ordered", "zone",
        "zones", "zoned", "throne", "thrones", "throned",
    }
    return _expand_group_verbs(
        ("govern.v.01", "rule.v.01", "command.v.01", "forbid.v.01"),
        _BASE_GROUP_COLLECTIVE_VERBS,
        blacklist,
    )


def _load_speech_verb_lemmas() -> frozenset[str]:
    """Build a narrow dialogue-tag verb set for quote attribution.

    The blacklist trims broader communication verbs whose common senses would
    make attribution too permissive. The fallback still includes the full
    manual seed set, so environments without WordNet keep stable behavior.
    """
    blacklist = {
        "communicate", "inform", "mention", "report", "broadcast",
        "phone", "sing", "chant", "read", "write",
    }
    return _expand_seeded_verbs(
        (
            "say.v.01",
            "ask.v.01",
            "whisper.v.01",
            "exclaim.v.01",
            "interject.v.01",
            "remark.v.01",
        ),
        _BASE_SPEECH_VERB_LEMMAS,
        blacklist,
    )


GROUP_MEMBERSHIP_VERBS: frozenset[str] = _load_group_membership_verbs()

GROUP_LEADERSHIP_NOUNS: frozenset[str] = _load_group_leadership_nouns()

GROUP_COLLECTIVE_VERBS: frozenset[str] = _load_group_collective_verbs()

SPEECH_VERB_LEMMAS: frozenset[str] = _load_speech_verb_lemmas()


def _load_relation_role_nouns() -> frozenset[str]:
    """Build a conservative kinship and relation-role noun set.

    The semantic-review layer needs recurring relation nouns such as
    "mother", "uncle", and "mentor", but not a broad bag of generic social
    nouns. WordNet is used as a narrow expansion source around kinship and
    guidance roots, with strict filtering and a hard fallback to the manual
    seed set.

    Returns:
        A frozenset of lowercase single-token relation-role nouns.
    """
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return _BASE_RELATION_ROLE_NOUNS

    root_synset_names = (
        "father.n.01",
        "mother.n.01",
        "parent.n.01",
        "sibling.n.01",
        "child.n.02",
        "spouse.n.01",
        "aunt.n.01",
        "uncle.n.01",
        "cousin.n.01",
    )
    blacklist = {
        "ancestor", "descendant", "kin", "relative", "relation", "family",
        "associate", "supporter", "friend", "lover", "mate", "hero",
        "guide", "teacher", "politician", "leader",
    }

    try:
        relation_words = set(_BASE_RELATION_ROLE_NOUNS)
        for name in root_synset_names:
            synset = wn.synset(name)
            candidate_synsets = [synset] + synset.hyponyms()
            for candidate in candidate_synsets:
                for lemma in candidate.lemmas():
                    word = lemma.name().lower()
                    if "_" in word or not word.isalpha():
                        continue
                    if len(word) < 3 or word in blacklist:
                        continue
                    relation_words.add(word)
        return frozenset(relation_words)
    except Exception:
        return _BASE_RELATION_ROLE_NOUNS


RELATION_ROLE_NOUNS: frozenset[str] = _load_relation_role_nouns()


def _load_event_nouns() -> frozenset[str]:
    """Build a conservative event-head noun set.

    Event classification needs common event nouns such as "festival",
    "procession", and "pageant", not a broad bag of abstract nouns. WordNet
    is used as a lexical expansion source from the manual seed words, with
    strict filtering and a hard fallback to the hand-curated list.

    Returns:
        A frozenset of lowercase single-token nouns that are plausible event
        heads in local contexts such as "during the Festival".
    """
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return _BASE_EVENT_NOUNS

    blacklist = {
        "event", "happening", "occurrence", "change", "activity", "act",
        "process", "development", "move", "motion", "cause", "effect",
        "experience", "case", "affair", "matter", "thing", "circumstance",
        "ceremonial", "occasion",
    }

    try:
        event_words = set(_BASE_EVENT_NOUNS)
        for seed_word in _BASE_EVENT_NOUNS:
            seed_synsets = [
                synset for synset in wn.synsets(seed_word, pos=wn.NOUN)
                if synset.lexname() == "noun.event"
            ]
            if not seed_synsets:
                continue

            seed_synset = seed_synsets[0]
            candidate_synsets = [seed_synset] + seed_synset.hyponyms()
            for candidate in candidate_synsets:
                for lemma in candidate.lemmas():
                    word = lemma.name().lower()
                    if "_" in word or not word.isalpha():
                        continue
                    if len(word) < 3 or word in blacklist:
                        continue
                    event_words.add(word)

        return frozenset(event_words)
    except Exception:
        return _BASE_EVENT_NOUNS


EVENT_NOUNS: frozenset[str] = _load_event_nouns()


def _load_place_descriptor_nouns() -> frozenset[str]:
    """Build a conservative place-descriptor noun set.

    The place classifier needs common nouns such as "city", "valley", and
    "seaport", not a gazetteer of actual named places. WordNet is therefore
    used only as a lexical expansion source from selected geographic root
    synsets, with strict filtering and a fallback to the hand-curated seed set.

    Returns:
        A frozenset of lowercase single-token common nouns that can act as
        place descriptors in local context such as "city of Sidhe".
    """
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return _BASE_PLACE_DESCRIPTOR_NOUNS

    root_synset_names = (
        "city.n.01",
        "town.n.01",
        "village.n.01",
        "municipality.n.01",
        "administrative_district.n.01",
        "district.n.01",
        "region.n.03",
        "geographical_area.n.01",
        "community.n.01",
        "province.n.01",
        "country.n.02",
        "continent.n.01",
        "island.n.01",
        "archipelago.n.01",
        "body_of_water.n.01",
        "river.n.01",
        "lake.n.01",
        "bay.n.01",
        "gulf.n.01",
        "mountain.n.01",
        "mountain_range.n.01",
        "valley.n.01",
        "forest.n.01",
        "woods.n.01",
        "plain.n.01",
        "plateau.n.01",
        "desert.n.01",
        "garden.n.01",
        "park.n.02",
        "port.n.01",
        "harbor.n.01",
        "station.n.01",
    )
    blacklist = {
        "common", "commons", "green", "field", "water", "land", "state",
        "range", "chain", "mount", "area", "community", "district",
        "territory", "dominion", "park",
    }

    try:
        descriptor_words = set(_BASE_PLACE_DESCRIPTOR_NOUNS)
        for name in root_synset_names:
            synset = wn.synset(name)
            candidate_synsets = [synset] + synset.hyponyms()
            for candidate in candidate_synsets:
                for lemma in candidate.lemmas():
                    word = lemma.name().lower()
                    if "_" in word or not word.isalpha():
                        continue
                    if len(word) < 3 or word in blacklist:
                        continue
                    descriptor_words.add(word)
        return frozenset(descriptor_words)
    except Exception:
        return _BASE_PLACE_DESCRIPTOR_NOUNS


PLACE_DESCRIPTOR_NOUNS: frozenset[str] = _load_place_descriptor_nouns()


def _load_place_possessive_context_nouns() -> frozenset[str]:
    """Build a conservative place-owned feature noun set.

    These nouns support patterns such as "Numazu's streets" and
    "Sidhe's harbor". The set should include civic or terrain features that
    plausibly belong to a place, but exclude indoor or personal-possession
    nouns like "room" and "house".

    Returns:
        A frozenset of lowercase single-token place-feature nouns.
    """
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return _BASE_PLACE_POSSESSIVE_CONTEXT_NOUNS

    root_synset_names = (
        "street.n.01",
        "road.n.01",
        "shore.n.01",
        "sand.n.01",
        "harbor.n.01",
        "bay.n.01",
        "beach.n.01",
        "forest.n.01",
        "garden.n.01",
        "port.n.01",
        "capital.n.03",
    )
    blacklist = {
        "room", "house", "home", "building", "door", "window", "bed",
        "desk", "office", "school", "campus", "yard", "field", "side",
        "edge", "corner", "property", "estate", "farm",
    }

    try:
        feature_words = set(_BASE_PLACE_POSSESSIVE_CONTEXT_NOUNS)
        for name in root_synset_names:
            synset = wn.synset(name)
            candidate_synsets = [synset] + synset.hyponyms()
            for candidate in candidate_synsets:
                for lemma in candidate.lemmas():
                    word = lemma.name().lower()
                    if "_" in word or not word.isalpha():
                        continue
                    if len(word) < 3 or word in blacklist:
                        continue
                    feature_words.add(word)
        return frozenset(feature_words)
    except Exception:
        return _BASE_PLACE_POSSESSIVE_CONTEXT_NOUNS


PLACE_POSSESSIVE_CONTEXT_NOUNS: frozenset[str] = _load_place_possessive_context_nouns()

PLACE_OF_CONTEXT_NOUNS: frozenset[str] = PLACE_DESCRIPTOR_NOUNS | frozenset({
    "heart", "center", "centre", "core", "edge",
})

# ---------------------------------------------------------------------------
# Faction suffixes
#
# Normalized-key suffixes that indicate a cluster names a group, organisation,
# or faction rather than an individual. Checked against the cluster's
# normalized_key in _assign_category. The list is intentionally broad to cover
# both realistic fiction and fantasy tropes.
# ---------------------------------------------------------------------------
FACTION_SUFFIXES: frozenset[str] = frozenset({
    # Formal organisations
    "council", "guild", "union", "assembly", "congress", "committee",
    "parliament", "senate", "court", "tribunal", "institute", "academy",
    "foundation", "bureau", "agency", "ministry", "department",
    # Military and paramilitary
    "order", "legion", "corps", "regiment", "battalion", "brigade",
    "cohort", "garrison", "vanguard", "guard", "watch",
    # Cultural and religious
    "clan", "tribe", "house", "brotherhood", "sisterhood", "fellowship",
    "covenant", "conclave", "chapter", "congregation", "sect", "cult",
    "circle", "coven", "lodge",
    # Fantasy tropes
    "alliance", "confederation", "empire", "realm", "dominion",
    "collective", "syndicate", "cabal",
})

# ---------------------------------------------------------------------------
# Field label list
#
# Used by structured-document harvesters to recognise labeled fields
# ("Alias: The Quiet One", "Faction: The Fleet").  Stored lowercase so
# harvesters can compare against token.text.lower().
# ---------------------------------------------------------------------------
FIELD_LABELS: frozenset[str] = frozenset({
    'alias', 'aliases',
    'role', 'roles',
    'faction', 'factions',
    'affiliation', 'affiliations',
    'occupation', 'job',
    'title', 'rank',
    'location', 'place', 'setting', 'homeworld', 'origin',
    'status', 'age', 'gender',
    'species', 'race',
    'weapon', 'weapons',
    'ability', 'abilities', 'skill', 'skills',
    'description', 'appearance', 'personality', 'motivation',
    'goal', 'goals',
    'relationship', 'relationships', 'family',
    'note', 'notes', 'misc', 'other',
})


# ---------------------------------------------------------------------------
# Stopwords
#
# Loaded from NLTK on first import. If the NLTK corpus is not available,
# a minimal fallback set is used so the pipeline degrades gracefully rather
# than crashing.  The fallback covers the most common sentence-initial words
# that would otherwise be mistaken for character names.
# ---------------------------------------------------------------------------

def _load_stopwords() -> frozenset[str]:
    """Load English stopwords from NLTK, downloading the corpus if needed.

    Returns:
        A frozenset of lowercase English stopwords.
    """
    try:
        from nltk.corpus import stopwords as _sw
        return frozenset(_sw.words('english'))
    except LookupError:
        pass

    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords as _sw
        return frozenset(_sw.words('english'))
    except Exception:
        pass

    return frozenset({
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'shall',
        'this', 'that', 'these', 'those', 'it', 'its', 'not', 'no',
        'up', 'out', 'so', 'if', 'as', 'then', 'than', 'he', 'she',
        'they', 'we', 'i', 'you', 'him', 'her', 'his', 'their', 'our',
        'me', 'my', 'your', 'who', 'what', 'which', 'when', 'where',
        'how', 'all', 'each', 'more', 'some', 'such', 'into', 'about',
    })


STOPWORDS: frozenset[str] = _load_stopwords()


def is_stopword(text: str) -> bool:
    """Return True if the token text is a stopword.

    Comparison is case-insensitive to handle sentence-initial capitals.

    Args:
        text: Token surface form (may be mixed case).

    Returns:
        True if the lowercased text appears in the stopword set.
    """
    return text.lower() in STOPWORDS


@lru_cache(maxsize=2048)
def has_generic_verb_sense(text: str) -> bool:
    """Return True when a lowercase token behaves like a generic verb lemma.

    This helper exists for late-stage suppression, not harvesting. Some words
    that survive NLTK stopwords are still ordinary prose verbs rather than
    plausible entity names. A word is treated as generic verb noise only when
    its WordNet verb inventory clearly outweighs its noun inventory. This
    keeps ambiguous name-like words such as "mark", "will", and "hope" out of
    the suppression bucket while still catching prose verbs such as "let",
    "tell", and "think".

    Args:
        text: Lowercase candidate token to inspect.

    Returns:
        True when the word has several verb senses and those senses dominate
        the noun inventory in WordNet.
    """
    word = text.lower()
    if not word.isalpha():
        return False

    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return False

    try:
        verb_synsets = wn.synsets(word, pos=wn.VERB)
        noun_synsets = wn.synsets(word, pos=wn.NOUN)
    except Exception:
        return False

    return len(verb_synsets) >= 3 and len(verb_synsets) > len(noun_synsets)


@lru_cache(maxsize=2048)
def has_generic_modifier_profile(text: str) -> bool:
    """Return True when a lowercase token behaves like a common modifier word.

    This helper is used only for late overlap suppression. A token counts as a
    generic modifier when WordNet treats it as an ordinary adjective or common
    noun with several broad senses. Proper-name-like tokens tend to have few or
    no such senses, which keeps person-name components such as "tsushima" from
    being collapsed just because they overlap with a longer compound.

    Args:
        text: Lowercase candidate token to inspect.

    Returns:
        True when the token looks like a broad adjective or common noun.
    """
    word = text.lower()
    if not word.isalpha():
        return False

    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return False

    try:
        adjective_synsets = wn.synsets(word, pos=wn.ADJ)
        noun_synsets = wn.synsets(word, pos=wn.NOUN)
    except Exception:
        return False

    return len(adjective_synsets) >= 3 or len(noun_synsets) >= 3


def normalize_surface(surface: str) -> str:
    """Compute the normalised clustering key for a surface form.

    Strips a leading title prefix and a trailing possessive suffix, then
    lowercases the result. This ensures that "Captain Aldous", "Aldous's",
    and "Aldous" all cluster under the same key "aldous".

    Args:
        surface: The raw surface form, e.g. "Captain Aldous's".

    Returns:
        Normalised form, e.g. "aldous".
    """
    parts = surface.split()
    if parts and parts[0] in TITLE_PREFIXES:
        parts = parts[1:]
    if not parts:
        return surface.lower().strip()
    name = ' '.join(parts)
    # Strip trailing possessive suffix.
    if name.endswith("'s"):
        name = name[:-2]
    elif name.endswith("s'"):
        name = name[:-1]
    return name.lower().strip()
