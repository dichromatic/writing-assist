# Diagram omitted - utility module with no significant information flow.

"""
Shared constants and utilities for all harvesting modules.

All title prefix lists, field labels, stopwords, and suppression helpers live
here. Harvesting modules import from this file only - never define their own
copies of these constants.
"""

from __future__ import annotations

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

    # Minimal fallback: covers the words most likely to appear sentence-initial
    # and be mistaken for proper names when capitalised.
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
