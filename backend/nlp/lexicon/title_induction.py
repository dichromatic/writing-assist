"""
Inductive title-prefix discovery from compound mention evidence.

.. code-block:: mermaid

    flowchart TD
        A[Final clusters + candidates + lexicon] --> B[Build eligible character keys]
        B --> C[Scan compound_capitalized candidates]
        C --> D[Collect leading-token support]
        D --> E[Apply support and safety filters]
        E --> F[Induced title prefixes + diagnostics]
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.nlp.harvesting.shared import TITLE_PREFIXES, TITLE_PREFIXES_LOWER, is_stopword
from backend.nlp.types import BootstrappedLexiconEntry, LexiconCategory, MentionCandidate, MentionCluster


@dataclass(frozen=True)
class TitleInductionDiagnostic:
    """Trace record for one title induction candidate token."""

    token: str
    token_lower: str
    support_count: int
    supporting_character_keys: list[str]
    accepted: bool
    reason: str


def induce_title_prefixes(
    *,
    clusters: list[MentionCluster],
    candidates: list[MentionCandidate],
    lexicon: list[BootstrappedLexiconEntry],
) -> tuple[frozenset[str], list[TitleInductionDiagnostic]]:
    """Infer title prefixes from compounds that consistently lead character keys.

    Args:
        clusters: Final mention clusters from bootstrap.
        candidates: Final mention candidates used for those clusters.
        lexicon: Final induced lexicon entries.

    Returns:
        Tuple of induced title prefixes and per-token diagnostics.
    """
    cluster_by_key = {cluster.normalized_key: cluster for cluster in clusters}
    character_keys: set[str] = set()
    for entry in lexicon:
        if entry.category != LexiconCategory.CHARACTER:
            continue
        cluster = cluster_by_key.get(entry.normalized_phrase)
        if cluster is None or cluster.occurrence_count < 3:
            continue
        character_keys.add(entry.normalized_phrase)
        # Also expose the last token of multi-token keys so that compounds like
        # "Director Watanabe" can match against "watanabe" even though the
        # canonical character key is "watanabe yō".
        key_parts = entry.normalized_phrase.split()
        if len(key_parts) > 1:
            last = key_parts[-1]
            if not is_stopword(last) and last not in TITLE_PREFIXES_LOWER:
                character_keys.add(last)

    standalone_counts: dict[str, int] = {}
    for cluster in clusters:
        standalone_counts[cluster.normalized_key] = cluster.occurrence_count

    supports: dict[str, set[str]] = {}
    token_case: dict[str, str] = {}
    token_hits: dict[str, int] = {}
    for candidate in candidates:
        if candidate.rule_source != "compound_capitalized":
            continue
        parts = candidate.surface.split()
        if len(parts) < 2:
            continue
        lead = parts[0]
        lead_lower = lead.lower()
        tail_key = " ".join(parts[1:]).lower()
        if lead_lower in TITLE_PREFIXES_LOWER:
            continue
        if tail_key not in character_keys:
            continue
        token_case.setdefault(lead_lower, lead)
        token_hits[lead_lower] = token_hits.get(lead_lower, 0) + 1
        supports.setdefault(lead_lower, set()).add(tail_key)

    induced: set[str] = set()
    diagnostics: list[TitleInductionDiagnostic] = []
    for token_lower in sorted(token_hits.keys()):
        support_count = token_hits[token_lower]
        supporting_character_keys = sorted(supports.get(token_lower, set()))
        token = token_case.get(token_lower, token_lower.title())

        if token in TITLE_PREFIXES:
            diagnostics.append(
                TitleInductionDiagnostic(
                    token=token,
                    token_lower=token_lower,
                    support_count=support_count,
                    supporting_character_keys=supporting_character_keys,
                    accepted=False,
                    reason="already_known_title_prefix",
                )
            )
            continue

        if support_count < 2:
            diagnostics.append(
                TitleInductionDiagnostic(
                    token=token,
                    token_lower=token_lower,
                    support_count=support_count,
                    supporting_character_keys=supporting_character_keys,
                    accepted=False,
                    reason="insufficient_support_count",
                )
            )
            continue

        standalone = standalone_counts.get(token_lower, 0)
        # Reject when the token appears far more often standalone than as a
        # title prefix. A genuine title like "Explorer" may appear standalone
        # many times as a rank reference, but if it also consistently precedes
        # character names the ratio will be close. A character name like
        # "Yoshiko" (200+ standalone, 0-1 as title prefix) is rejected here.
        # The multiplier of 4 means: standalone must be less than 4x the
        # compound evidence count to pass.
        if standalone >= 5 and standalone > support_count * 4:
            diagnostics.append(
                TitleInductionDiagnostic(
                    token=token,
                    token_lower=token_lower,
                    support_count=support_count,
                    supporting_character_keys=supporting_character_keys,
                    accepted=False,
                    reason="strong_standalone_key",
                )
            )
            continue

        induced.add(token)
        diagnostics.append(
            TitleInductionDiagnostic(
                token=token,
                token_lower=token_lower,
                support_count=support_count,
                supporting_character_keys=supporting_character_keys,
                accepted=True,
                reason="accepted",
            )
        )

    return frozenset(induced), diagnostics

