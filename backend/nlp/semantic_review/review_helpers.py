"""
Semantic review helpers - cluster references and prepare review context.

.. code-block:: mermaid

    flowchart TD
        A[ReferenceCandidate list] --> B[Build ReferenceCluster list]
        C[CorpusEntity list] --> D[Build ConflictRecord list]
        B & D --> E[Build CharacterSemanticSummary list]
        B & E --> F[Build ReviewContext]
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictRecord,
    ConflictSource,
    CorpusEntity,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityRecord,
    LexiconCategory,
    ReferenceCandidate,
    ReferenceCandidateType,
    ReferenceCluster,
    SpanAnchor,
    SuppressedEvidence,
)

_REFERENCE_SUPPRESSED_ORBIT_CHARS = 40


@dataclass(frozen=True)
class ReviewContext:
    """Precomputed ownership maps shared by semantic proposals and tasks.

    Args:
        ownership_mode: Whether the context was built from raw reference
            clusters or canonical character summaries.
        title_owner_scores: Ranked title-owner counts by normalized title.
        relation_owner_scores: Ranked relation-owner counts by normalized
            relation noun.
        canonical_map: Alias-to-canonical lookup used to normalize local keys.
    """

    ownership_mode: Literal["reference_clusters", "canonical_character_summaries"]
    title_owner_scores: dict[str, dict[str, int]]
    relation_owner_scores: dict[str, dict[str, int]]
    canonical_map: dict[str, str]


def _rank_cluster_candidate_scores(
    candidate_entity_scores: dict[str, int],
    speaker_entity_scores: dict[str, int],
    address_like_count: int,
) -> dict[str, int]:
    """Return candidate scores in presentation order for semantic review.

    Address-like bare references often name the addressee rather than the
    speaker. When the current evidence says "this looks like direct address"
    and also knows who was speaking, the review surface should not present the
    speaker as the best target by default unless no better alternative exists.

    Args:
        candidate_entity_scores: Raw candidate target counts for a cluster.
        speaker_entity_scores: Counts of quote speakers observed for the same
            cluster.
        address_like_count: Number of address-like mentions in the cluster.

    Returns:
        A dict with stable insertion order that reflects review-time ranking.
    """
    if not candidate_entity_scores:
        return {}

    ranked_items = sorted(
        candidate_entity_scores.items(),
        key=lambda item: (
            address_like_count > 0 and item[0] in speaker_entity_scores,
            -item[1],
            -speaker_entity_scores.get(item[0], 0),
            item[0],
        ),
    )
    return dict(ranked_items)


def _suppressed_reference_distance(
    reference_anchor: SpanAnchor,
    suppressed_anchor: SpanAnchor,
) -> int | None:
    """Return local orbit distance for a reference and suppressed anchor.

    A suppressed item belongs in a reference cluster's orbit only when the two
    anchors live in the same span. Overlap counts as distance zero; otherwise
    the gap between the anchor boundaries is used.

    Args:
        reference_anchor: Anchor from a grouped semantic reference.
        suppressed_anchor: Anchor from a suppressed document-local record.

    Returns:
        The character gap between the anchors, or None when they are in
        different spans or documents.
    """
    if (
        reference_anchor.path != suppressed_anchor.path
        or reference_anchor.span_ordinal != suppressed_anchor.span_ordinal
    ):
        return None
    if reference_anchor.start_char < suppressed_anchor.end_char and suppressed_anchor.start_char < reference_anchor.end_char:
        return 0
    if suppressed_anchor.end_char <= reference_anchor.start_char:
        return reference_anchor.start_char - suppressed_anchor.end_char
    return suppressed_anchor.start_char - reference_anchor.end_char


def _attach_suppressed_evidence_to_reference_clusters(
    reference_clusters: list[ReferenceCluster],
    records: list[DocumentEntityRecord],
) -> None:
    """Attach nearby suppressed evidence to grouped reference clusters.

    Later semantic review benefits from seeing weak lexical debris, compound
    fragments, and suppressed title-like surfaces in the orbit of the
    reference they surrounded. This helper keeps that orbit rule narrow and
    deterministic by requiring same-span proximity.

    Args:
        reference_clusters: Grouped reference clusters to enrich in place.
        records: Document-local entity records for the same corpus slice.
    """
    suppressed_by_path: dict[str, list[DocumentEntityRecord]] = defaultdict(list)
    for record in records:
        if record.bucket == DocumentEntityBucket.SUPPRESSED:
            suppressed_by_path[record.document_anchor.path].append(record)

    for cluster in reference_clusters:
        nearby: list[tuple[int, SuppressedEvidence]] = []
        seen_keys: set[str] = set()
        for suppressed_record in suppressed_by_path.get(cluster.document_anchor.path, []):
            assert suppressed_record.suppression_reason is not None
            best_distance: int | None = None
            for reference_anchor in cluster.anchors:
                for suppressed_anchor in suppressed_record.anchors:
                    distance = _suppressed_reference_distance(reference_anchor, suppressed_anchor)
                    if distance is None or distance > _REFERENCE_SUPPRESSED_ORBIT_CHARS:
                        continue
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
            if best_distance is None:
                continue
            if suppressed_record.normalized_key in seen_keys:
                continue
            seen_keys.add(suppressed_record.normalized_key)
            nearby.append((best_distance, SuppressedEvidence(
                document_anchor=suppressed_record.document_anchor,
                normalized_key=suppressed_record.normalized_key,
                surface_forms=list(suppressed_record.surface_forms),
                winning_category=suppressed_record.winning_category,
                confidence_score=suppressed_record.confidence_score,
                reason=suppressed_record.suppression_reason,
                detail=suppressed_record.bucket_detail,
                anchors=list(suppressed_record.anchors),
            )))

        cluster.suppressed_related_evidence = [
            evidence for _distance, evidence in sorted(
                nearby,
                key=lambda item: (item[0], item[1].normalized_key),
            )
        ]


def build_conflict_records(entities: list[CorpusEntity]) -> list[ConflictRecord]:
    """Convert review-required corpus entities into typed conflict records.

    Args:
        entities: Corpus canonical entities from reconciliation.

    Returns:
        ConflictRecord entries for every review-required cross-category entity.
    """
    conflicts: list[ConflictRecord] = []
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        if not entity.review_required or not entity.conflicting_categories:
            continue

        canonical_categories = {
            record.winning_category
            for record in entity.member_records
            if record.resolved
            and record.winning_category != LexiconCategory.UNRESOLVED
            and record.normalized_key == entity.canonical_key
        }
        absorbed_categories = {
            record.winning_category
            for record in entity.member_records
            if record.resolved
            and record.winning_category != LexiconCategory.UNRESOLVED
            and record.normalized_key != entity.canonical_key
        }

        if (
            (canonical_categories and absorbed_categories - canonical_categories)
            or (not canonical_categories and absorbed_categories and len(entity.source_keys) > 1)
        ):
            source = ConflictSource.COMPONENT_POLLUTION
            reason = (
                f"absorbed alias components introduce conflicting categories for "
                f"'{entity.canonical_key}'"
            )
        else:
            source = ConflictSource.SURFACE_LEVEL_DISAGREEMENT
            reason = (
                f"exact surface evidence for '{entity.canonical_key}' disagrees across documents"
            )

        conflicts.append(ConflictRecord(
            canonical_key=entity.canonical_key,
            source=source,
            conflicting_categories=entity.conflicting_categories,
            supporting_document_paths=entity.supporting_document_paths,
            reason=reason,
        ))

    return conflicts


def build_reference_clusters(
    references: list[ReferenceCandidate],
    records: list[DocumentEntityRecord] | None = None,
) -> list[ReferenceCluster]:
    """Group repeated reference candidates into stable document-level clusters.

    Args:
        references: Raw reference candidates extracted from one or more
            documents.
        records: Optional document-local entity records used to attach nearby
            suppressed evidence to the grouped reference orbit.

    Returns:
        Grouped reference clusters in document and subtype order.
    """
    grouped: dict[tuple[str, str, str], list[ReferenceCandidate]] = defaultdict(list)
    bound_entity_scores: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for reference in references:
        grouped[(
            reference.document_anchor.path,
            reference.reference_type.value,
            reference.normalized,
        )].append(reference)
        if reference.reference_type in {
            ReferenceCandidateType.BOUND_TITLE_ROLE,
            ReferenceCandidateType.BOUND_RELATION_ROLE,
        }:
            for key in reference.linked_entity_keys:
                bound_entity_scores[(reference.document_anchor.path, reference.normalized)][key] += 1

    clusters: list[ReferenceCluster] = []
    for (_path, _rtype, _normalized), members in sorted(grouped.items()):
        candidate_entity_scores: dict[str, int] = defaultdict(int)
        speaker_entity_scores: dict[str, int] = defaultdict(int)
        for member in members:
            for key in member.linked_entity_keys:
                candidate_entity_scores[key] += 1
            if member.quote_speaker_key is not None:
                speaker_entity_scores[member.quote_speaker_key] += 1
        if members[0].reference_type in {
            ReferenceCandidateType.BARE_TITLE_ROLE,
            ReferenceCandidateType.BARE_RELATION_ROLE,
        }:
            bound_scores = bound_entity_scores[(members[0].document_anchor.path, members[0].normalized)]
            if not candidate_entity_scores and len(bound_scores) == 1:
                only_key = next(iter(bound_scores))
                candidate_entity_scores[only_key] = len(members)
        ranked_speaker_scores = dict(sorted(
            speaker_entity_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ))
        ranked_candidate_scores = _rank_cluster_candidate_scores(
            dict(candidate_entity_scores),
            ranked_speaker_scores,
            sum(1 for member in members if member.address_like),
        )
        clusters.append(ReferenceCluster(
            document_anchor=members[0].document_anchor,
            reference_type=members[0].reference_type,
            normalized=members[0].normalized,
            surface_forms=sorted({member.surface for member in members}),
            occurrence_count=len(members),
            anchors=sorted(
                [member.anchor for member in members],
                key=lambda anchor: (anchor.path, anchor.span_ordinal, anchor.start_char),
            ),
            in_quote_count=sum(1 for member in members if member.in_quote),
            address_like_count=sum(1 for member in members if member.address_like),
            speaker_entity_scores=ranked_speaker_scores,
            candidate_entity_scores=ranked_candidate_scores,
        ))

    if records is not None:
        _attach_suppressed_evidence_to_reference_clusters(clusters, records)

    return clusters


def build_character_summaries(
    entities: list[CorpusEntity],
    reference_clusters: list[ReferenceCluster],
    conflicts: list[ConflictRecord],
) -> list[CharacterSemanticSummary]:
    """Build character-centric semantic summaries from corpus review evidence.

    The deterministic pipeline already knows a lot about recurring characters,
    but that evidence is spread across entity aliases, grouped title mentions,
    and typed conflicts. This pass gathers those signals into one stable
    review object per canonical character so later semantic review can reason
    per person instead of scanning flat corpus tables.

    Args:
        entities: Corpus canonical entities from reconciliation.
        reference_clusters: Grouped semantic reference candidates.
        conflicts: Typed conflict records already built for the corpus.

    Returns:
        One summary per canonical character in stable presentation order.
    """
    character_entities = [
        entity for entity in entities
        if entity.dominant_category == LexiconCategory.CHARACTER
    ]
    character_keys_by_source: dict[str, str] = {}
    for entity in character_entities:
        for key in entity.source_keys:
            character_keys_by_source[key] = entity.canonical_key

    attached_title_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ambiguous_title_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    attached_relation_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ambiguous_relation_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cluster in reference_clusters:
        canonical_targets = sorted({
            character_keys_by_source[key]
            for key in cluster.candidate_entity_scores
            if key in character_keys_by_source
        })
        if cluster.reference_type in {
            ReferenceCandidateType.BOUND_TITLE_ROLE,
            ReferenceCandidateType.BARE_TITLE_ROLE,
        }:
            attached_counts = attached_title_counts
            ambiguous_counts = ambiguous_title_counts
        elif cluster.reference_type in {
            ReferenceCandidateType.BOUND_RELATION_ROLE,
            ReferenceCandidateType.BARE_RELATION_ROLE,
        }:
            attached_counts = attached_relation_counts
            ambiguous_counts = ambiguous_relation_counts
        else:
            continue
        if len(canonical_targets) == 1:
            attached_counts[canonical_targets[0]][cluster.normalized] += cluster.occurrence_count
        elif len(canonical_targets) > 1:
            for canonical_key in canonical_targets:
                ambiguous_counts[canonical_key][cluster.normalized] += cluster.occurrence_count

    conflicts_by_key: dict[str, list[ConflictSource]] = defaultdict(list)
    for conflict in conflicts:
        conflicts_by_key[conflict.canonical_key].append(conflict.source)

    summaries: list[CharacterSemanticSummary] = []
    for entity in sorted(
        character_entities,
        key=lambda item: (-len(item.supporting_document_paths), item.canonical_key),
    ):
        summaries.append(CharacterSemanticSummary(
            canonical_key=entity.canonical_key,
            alias_keys=[key for key in entity.source_keys if key != entity.canonical_key],
            canonical_surface_forms=entity.canonical_surface_forms,
            absorbed_surface_forms=entity.absorbed_surface_forms,
            supporting_document_paths=entity.supporting_document_paths,
            attached_title_counts=dict(sorted(
                attached_title_counts[entity.canonical_key].items(),
                key=lambda item: (-item[1], item[0]),
            )),
            ambiguous_title_counts=dict(sorted(
                ambiguous_title_counts[entity.canonical_key].items(),
                key=lambda item: (-item[1], item[0]),
            )),
            attached_relation_counts=dict(sorted(
                attached_relation_counts[entity.canonical_key].items(),
                key=lambda item: (-item[1], item[0]),
            )),
            ambiguous_relation_counts=dict(sorted(
                ambiguous_relation_counts[entity.canonical_key].items(),
                key=lambda item: (-item[1], item[0]),
            )),
            aggregate_attribution_count=sum(
                record.attribution_count for record in entity.member_records
            ),
            conflict_sources=sorted(
                conflicts_by_key[entity.canonical_key],
                key=lambda source: source.value,
            ),
            merge_reasons=list(entity.reasons),
        ))

    return summaries


def _build_title_owner_scores(
    reference_clusters: list[ReferenceCluster],
) -> dict[str, dict[str, int]]:
    """Aggregate recurring title ownership hints across the current corpus.

    Unique title attachments in one document are useful fallback evidence for
    harder address-like uses in another. This helper stays conservative by
    counting only title clusters that already point to exactly one character.

    Args:
        reference_clusters: Grouped semantic reference candidates.

    Returns:
        Mapping of normalized title to per-character evidence counts.
    """
    title_owner_scores: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cluster in reference_clusters:
        if cluster.reference_type not in {
            ReferenceCandidateType.BOUND_TITLE_ROLE,
            ReferenceCandidateType.BARE_TITLE_ROLE,
        }:
            continue
        if len(cluster.candidate_entity_scores) != 1:
            continue
        only_key = next(iter(cluster.candidate_entity_scores))
        title_owner_scores[cluster.normalized][only_key] += cluster.occurrence_count

    return {
        normalized: dict(sorted(
            owner_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ))
        for normalized, owner_scores in title_owner_scores.items()
    }


def _build_canonical_owner_scores(
    character_summaries: list[CharacterSemanticSummary],
    counts_getter: Callable[[CharacterSemanticSummary], dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Aggregate canonical ownership counts from character summaries.

    Character summaries already fold aliases into one canonical person, so
    title and relation counts derived from them are better fallback evidence
    than raw document-level keys when a review prompt or proposal needs corpus
    ownership hints.

    Args:
        character_summaries: Character-centric semantic summaries.
        counts_getter: Accessor that selects which attached-count map should
            contribute ownership evidence for each summary.

    Returns:
        Mapping of normalized owner label to canonical owner counts.
    """
    owner_scores: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for summary in character_summaries:
        for normalized, count in counts_getter(summary).items():
            owner_scores[normalized][summary.canonical_key] += count

    return {
        normalized: dict(sorted(
            owner_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ))
        for normalized, owner_scores in owner_scores.items()
    }


def _build_character_canonical_map(
    character_summaries: list[CharacterSemanticSummary],
) -> dict[str, str]:
    """Map character aliases back to canonical character keys."""
    canonical_map: dict[str, str] = {}
    for summary in character_summaries:
        canonical_map[summary.canonical_key] = summary.canonical_key
        for alias_key in summary.alias_keys:
            canonical_map[alias_key] = summary.canonical_key
    return canonical_map


def build_review_context(
    reference_clusters: list[ReferenceCluster],
    character_summaries: list[CharacterSemanticSummary] | None = None,
) -> ReviewContext:
    """Precompute reusable ownership maps for semantic review consumers.

    The semantic proposal and review-task builders read the same derived
    ownership structures. Making that shared state explicit avoids duplicate
    computation and, more importantly, records whether those structures came
    from raw reference clusters or canonical character summaries.

    Args:
        reference_clusters: Grouped semantic reference candidates.
        character_summaries: Optional canonical character summaries. When
            provided, canonical ownership maps are built from them.

    Returns:
        A ReviewContext carrying the precomputed ownership maps and source
        mode.
    """
    if character_summaries is not None:
        return ReviewContext(
            ownership_mode="canonical_character_summaries",
            title_owner_scores=_build_canonical_owner_scores(
                character_summaries,
                lambda summary: summary.attached_title_counts,
            ),
            relation_owner_scores=_build_canonical_owner_scores(
                character_summaries,
                lambda summary: summary.attached_relation_counts,
            ),
            canonical_map=_build_character_canonical_map(character_summaries),
        )

    return ReviewContext(
        ownership_mode="reference_clusters",
        title_owner_scores=_build_title_owner_scores(reference_clusters),
        relation_owner_scores={},
        canonical_map={},
    )


def _strong_owner_keys(owner_scores: dict[str, int]) -> list[str]:
    """Return only the strongest ownership candidates for review prompts.

    Corpus-level fallback should help a human or later semantic pass focus on
    plausible owners, not enumerate every weak one-off holder of a title.

    Args:
        owner_scores: Ranked owner counts for one normalized title.

    Returns:
        A short list of the strongest owner keys.
    """
    if not owner_scores:
        return []

    top_score = max(owner_scores.values())
    threshold = max(2, (top_score + 1) // 2)
    strong_keys = [
        key for key, score in owner_scores.items()
        if score >= threshold
    ]
    return strong_keys[:3]


def _dominant_owner_key(owner_scores: dict[str, int]) -> str | None:
    """Return one owner when corpus evidence clearly favors a single target.

    Args:
        owner_scores: Ranked owner counts for one normalized title.

    Returns:
        The dominant owner key when the top owner clearly separates from the
        runner-up, otherwise None.
    """
    if not owner_scores:
        return None

    ranked_items = list(owner_scores.items())
    top_key, top_score = ranked_items[0]
    if top_score < 3:
        return None
    if len(ranked_items) == 1:
        return top_key

    second_score = ranked_items[1][1]
    if top_score >= second_score * 2:
        return top_key
    return None


def _canonicalize_ranked_keys(
    keys: list[str],
    canonical_map: dict[str, str],
) -> list[str]:
    """Collapse alias keys to canonical character keys in stable order."""
    canonical_keys: list[str] = []
    seen: set[str] = set()
    for key in keys:
        canonical_key = canonical_map.get(key, key)
        if canonical_key in seen:
            continue
        seen.add(canonical_key)
        canonical_keys.append(canonical_key)
    return canonical_keys


def _reference_resolution_context(
    reference: ReferenceCluster,
    title_owner_scores: dict[str, dict[str, int]],
    relation_owner_scores: dict[str, dict[str, int]],
    canonical_map: dict[str, str],
) -> dict[str, object]:
    """Compute ranked local and corpus owner evidence for one reference."""
    ranked_entity_keys = _canonicalize_ranked_keys(
        list(reference.candidate_entity_scores.keys()),
        canonical_map,
    )
    ranked_speaker_keys = _canonicalize_ranked_keys(
        list(reference.speaker_entity_scores.keys()),
        canonical_map,
    )
    non_speaker_entity_keys = [
        key for key in ranked_entity_keys
        if key not in ranked_speaker_keys
    ]

    corpus_owner_keys: list[str] = []
    dominant_owner_key: str | None = None
    if reference.reference_type == ReferenceCandidateType.BARE_TITLE_ROLE:
        non_speaker_owner_scores = {
            key: score
            for key, score in title_owner_scores.get(reference.normalized, {}).items()
            if key not in ranked_speaker_keys
        }
        corpus_owner_keys = _strong_owner_keys(non_speaker_owner_scores)
        dominant_owner_key = _dominant_owner_key(non_speaker_owner_scores)
    elif reference.reference_type == ReferenceCandidateType.BARE_RELATION_ROLE:
        non_speaker_owner_scores = {
            key: score
            for key, score in relation_owner_scores.get(reference.normalized, {}).items()
            if key not in ranked_speaker_keys
        }
        corpus_owner_keys = _strong_owner_keys(non_speaker_owner_scores)
        dominant_owner_key = _dominant_owner_key(non_speaker_owner_scores)

    return {
        "ranked_entity_keys": ranked_entity_keys,
        "ranked_speaker_keys": ranked_speaker_keys,
        "non_speaker_entity_keys": non_speaker_entity_keys,
        "corpus_owner_keys": corpus_owner_keys,
        "dominant_owner_key": dominant_owner_key,
    }


def _reference_evidence_note(
    reference: ReferenceCluster,
    ranked_entity_keys: list[str],
    ranked_speaker_keys: list[str],
    corpus_owner_keys: list[str],
    dominant_owner_key: str | None,
) -> str:
    """Summarize why one reference keeps ranked alternatives in handoff.

    The manuscript handoff should preserve how deterministic ranking was
    formed without pretending the ranking is semantic truth. A short structured
    note keeps later review grounded in evidence type rather than only reading
    the generated prompt text.

    Args:
        reference: Grouped semantic reference under review.
        ranked_entity_keys: Ranked local deterministic candidates.
        ranked_speaker_keys: Ranked quote speakers for the grouped mentions.
        corpus_owner_keys: Strong recurring corpus owners for the same label.
        dominant_owner_key: One clearly dominant owner, when corpus evidence
            strongly favors it.

    Returns:
        A short explanation of the retained ranking evidence.
    """
    evidence_parts: list[str] = []
    if ranked_entity_keys:
        evidence_parts.append("local candidates preserved")
    if ranked_speaker_keys:
        evidence_parts.append("quote speakers preserved")
    if reference.address_like_count:
        evidence_parts.append("address-like usage lowers speaker priority")
    if dominant_owner_key is not None:
        evidence_parts.append("one dominant corpus owner preserved")
    elif corpus_owner_keys:
        evidence_parts.append("strong corpus owners preserved")
    if not evidence_parts:
        evidence_parts.append("no deterministic target ranking available")
    return "; ".join(evidence_parts)


def _reference_review_prompt(
    reference: ReferenceCluster,
    label: str,
    ranked_entity_keys: list[str],
    ranked_speaker_keys: list[str],
    non_speaker_entity_keys: list[str],
    corpus_owner_keys: list[str],
    dominant_owner_key: str | None,
) -> str:
    """Build the exact review prompt text for one deferred reference.

    The prompt string is not presentation-only. It participates in the task
    dedupe key, so this helper must preserve the existing branch priority and
    byte-level wording of the emitted prompt variants.

    Args:
        reference: Grouped semantic reference under review.
        label: Human-readable role label such as ``title`` or ``relation``.
        ranked_entity_keys: Ranked local deterministic candidates.
        ranked_speaker_keys: Ranked quote speakers for the grouped mentions.
        non_speaker_entity_keys: Ranked local candidates excluding speakers.
        corpus_owner_keys: Strong recurring corpus owners for the same label.
        dominant_owner_key: One clearly dominant owner, when present.

    Returns:
        The stable prompt text used for review-task emission and dedupe.
    """
    speaker_text = ""
    if reference.address_like_count and ranked_speaker_keys:
        speaker_text = f" spoken by {', '.join(ranked_speaker_keys)}"

    if reference.address_like_count and ranked_speaker_keys and non_speaker_entity_keys:
        return (
            f"Does the address-like bare {label} '{reference.normalized}'"
            f"{speaker_text} refer to one of {', '.join(non_speaker_entity_keys)}?"
        )

    if (
        reference.address_like_count
        and ranked_speaker_keys
        and dominant_owner_key is not None
    ):
        return (
            f"Does the address-like bare {label} '{reference.normalized}'"
            f"{speaker_text} most likely refer to {dominant_owner_key}?"
        )

    if (
        reference.address_like_count
        and ranked_speaker_keys
        and corpus_owner_keys
    ):
        return (
            f"Does the address-like bare {label} '{reference.normalized}'"
            f"{speaker_text} refer to one of {', '.join(corpus_owner_keys)}?"
        )

    if reference.address_like_count and ranked_speaker_keys and ranked_entity_keys:
        return (
            f"Does the address-like bare {label} '{reference.normalized}'"
            f"{speaker_text} refer to a recurring character other than the speaker?"
        )

    if ranked_entity_keys:
        return (
            f"Does the "
            f"{'address-like ' if reference.address_like_count else ''}"
            f"bare {label} '{reference.normalized}'{speaker_text} refer to one of "
            f"{', '.join(ranked_entity_keys)}?"
        )

    return (
        f"Does the "
        f"{'address-like ' if reference.address_like_count else ''}"
        f"bare {label} '{reference.normalized}'{speaker_text} refer to a recurring "
        f"character or role in this document?"
    )
