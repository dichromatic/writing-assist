"""
Suppression rescue task builder - select rescue candidates and assemble task packets.

The deterministic extraction pipeline suppresses ~800+ entities. Most
suppressions are correct, but a subset are real entities that fail structural
heuristics. This module selects plausible rescue candidates and builds LLM
task packets for binary verification.

.. code-block:: mermaid

    flowchart TD
        A[ManuscriptReviewBundle] --> B[Filter suppressed entity records]
        B --> C{Rescue candidate?}
        C -->|Yes| D[Build evidence from raw text]
        C -->|No| E[Diagnostic: rejected]
        D --> F[LLMTaskPacket]
        E --> G[LLMTaskSelectionDiagnostic]
        F & G --> H[Packet + diagnostic lists]
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.nlp.types import (
    DocumentEntityBucket,
    DocumentEntityRecord,
    DocumentStatus,
    DocumentType,
    LLMTaskEvidenceItem,
    LLMTaskFamily,
    LLMTaskPacket,
    LLMTaskSelectionDiagnostic,
    LexiconCategory,
    ManuscriptReviewBundle,
    SpanAnchor,
    SuppressReason,
    stable_hash_id,
)

_RESCUE_CONTEXT_RADIUS = 300
_RESCUE_EVIDENCE_LIMIT = 5

_RESCUABLE_REASONS = {
    SuppressReason.LOW_ENTITYHOOD,
    SuppressReason.COMPONENT_OVERLAP_NOISE,
    SuppressReason.GENERIC_LEXICAL_NOISE,
}

_SCHEMA_ID = "manuscript_suppression_rescue.v1"


def _discourse_rescue_rejection_reason(
    record: DocumentEntityRecord,
) -> str | None:
    """Return a discourse-based rejection reason when the usage pattern is junk-like.

    Quote-only unresolved tokens that appear mainly as address-like dialogue
    debris or one-token utterances are the clearest low-value rescue calls in
    the current manuscript corpus. This gate removes those without relying on
    project-specific word lists.
    """
    profile = record.discourse_profile

    if record.current_state.winning_category != LexiconCategory.UNRESOLVED:
        return None
    if not profile.quote_only or profile.non_quote_count > 0:
        return None
    if profile.address_like_count > 0:
        return "quote_only_address_like_discourse"
    if profile.one_token_utterance_count > 0:
        return "quote_only_one_token_discourse"
    return None


def _lineage_rescue_rejection_reason(
    record: DocumentEntityRecord,
    surviving_compound_keys: frozenset[str],
) -> str | None:
    """Return a lineage-based rejection reason for structurally dependent fragments.

    Two layers of lineage rejection apply:

    1. Component-overlap fragments (suppressed as COMPONENT_OVERLAP_NOISE) are
       rejected when fully covered by longer compounds with no uncovered anchors.
    2. Any suppressed record whose every occurrence is explained by a surviving
       compound (promoted or review_only) is rejected as structurally redundant,
       regardless of suppression reason.

    The second layer catches cases like "archive" (suppressed as low_entityhood)
    that are fully covered by the surviving compound "archive wing".
    """
    profile = record.lineage_profile

    # Layer 1: component-overlap-specific checks (original gate).
    if record.promotion_trace.suppression_reason == SuppressReason.COMPONENT_OVERLAP_NOISE:
        if profile.fully_covered_by_longer_compound and profile.uncovered_anchor_count == 0:
            return "fully_covered_by_longer_compound"
        if profile.appears_as_compound_component and profile.uncovered_anchor_count == 0:
            return "component_only_no_uncovered_support"

    # Layer 2: component of a surviving compound with no independent usage.
    if (
        profile.appears_as_compound_component
        and profile.uncovered_anchor_count == 0
        and any(pk in surviving_compound_keys for pk in profile.candidate_parent_keys)
    ):
        return "component_of_surviving_compound"

    return None


def _rescue_candidate_selected(
    record: DocumentEntityRecord,
    absorbed_keys: frozenset[str],
    surviving_compound_keys: frozenset[str],
) -> tuple[bool, str]:
    """Return whether one suppressed record qualifies for LLM rescue triage.

    The gate narrows ~800+ suppressed records to ~30-50 candidates by
    requiring a rescuable suppression reason, sufficient occurrences, and
    no prior absorption into a compound entity.
    """
    if record.current_state.bucket != DocumentEntityBucket.SUPPRESSED:
        return False, "not_suppressed"
    if record.identity.normalized_key in absorbed_keys:
        return False, "already_absorbed_into_compound"
    if record.promotion_trace.suppression_reason not in _RESCUABLE_REASONS:
        reason = record.promotion_trace.suppression_reason.value if record.promotion_trace.suppression_reason is not None else "none"
        return False, f"suppression_reason_{reason}_not_rescuable"
    # The group-level positive-signal gate is now the primary quality
    # filter. The per-record occurrence threshold is kept only as a
    # minimal noise floor: single-occurrence records in a document are
    # too sparse to build useful evidence windows for the LLM.
    if record.source_evidence.occurrence_count < 1:
        return False, "too_few_occurrences"

    # Single-scene entities are allowed through. Characters in flashback
    # chapters or ships mentioned only in one scene still deserve LLM triage.
    if (
        record.current_state.winning_category == LexiconCategory.UNRESOLVED
        and record.classification_trace.entityhood.score < 0.25
    ):
        return False, "unresolved_very_low_entityhood"

    discourse_rejection = _discourse_rescue_rejection_reason(record)
    if discourse_rejection is not None:
        return False, discourse_rejection

    lineage_rejection = _lineage_rescue_rejection_reason(record, surviving_compound_keys)
    if lineage_rejection is not None:
        return False, lineage_rejection

    return True, "rescue_candidate"


def _build_evidence_item(
    *,
    source_object_id: str,
    anchor: SpanAnchor,
    quote: str,
    context_before: str,
    context_after: str,
    suppression_reason: str,
    confidence_score: float | None,
) -> LLMTaskEvidenceItem:
    """Build one evidence item for a rescue task packet."""
    return LLMTaskEvidenceItem(
        evidence_id=stable_hash_id(
            "llm_task_evidence",
            source_object_id,
            anchor.path,
            str(anchor.start_char),
            str(anchor.end_char),
            quote,
        ),
        document_path=anchor.path,
        source_anchor=anchor,
        quote=quote,
        context_before=context_before,
        context_after=context_after,
        source_object_id=source_object_id,
        visibility_bucket="suppressed",
        suppression_reason=suppression_reason,
        confidence_score=confidence_score,
    )


def _build_rescue_evidence(
    record: DocumentEntityRecord,
    raw_text: str,
) -> list[LLMTaskEvidenceItem]:
    """Build evidence windows from anchors and raw document text.

    Slices context around each anchor position at assembly time so
    suppressed records do not need promotion-stage context windows.
    """
    items: list[LLMTaskEvidenceItem] = []
    sorted_anchors = sorted(record.source_evidence.anchors, key=lambda a: a.start_char)
    quote = record.identity.surface_forms[0] if record.identity.surface_forms else record.identity.normalized_key
    suppression = record.promotion_trace.suppression_reason.value if record.promotion_trace.suppression_reason else ""

    for anchor in sorted_anchors[:_RESCUE_EVIDENCE_LIMIT]:
        before_start = max(0, anchor.start_char - _RESCUE_CONTEXT_RADIUS)
        after_end = min(len(raw_text), anchor.end_char + _RESCUE_CONTEXT_RADIUS)
        context_before = raw_text[before_start:anchor.start_char]
        context_after = raw_text[anchor.end_char:after_end]
        if not context_before.strip() and not context_after.strip():
            continue
        items.append(
            _build_evidence_item(
                source_object_id=record.identity.normalized_key,
                anchor=anchor,
                quote=quote,
                context_before=context_before,
                context_after=context_after,
                suppression_reason=suppression,
                confidence_score=record.promotion_trace.confidence_score,
            )
        )
    return items


@dataclass
class _RescueGroup:
    """Accumulator for merging multiple document-level records of one entity."""

    normalized_key: str
    records: list[DocumentEntityRecord]
    evidence: list[LLMTaskEvidenceItem]
    document_paths: list[str]
    surface_forms: set[str]
    total_occurrences: int
    total_scenes: int
    suppression_reasons: set[str]
    winning_categories: set[str]
    best_confidence: float
    best_entityhood: float
    best_title_support: int
    best_possessive_support: int
    best_location_support: int
    best_linked_field: int
    best_linked_definition: int
    best_linked_seed: int
    max_uncovered_anchors: int
    has_independent_overlap_fragment: bool
    has_compound_participation: bool
    cross_doc_total_occurrences: int


def _rescue_group_has_positive_signal(group: _RescueGroup) -> tuple[bool, str]:
    """Return whether a rescue group has at least one positive deterministic signal.

    A positive signal distinguishes genuinely ambiguous suppressed entities from
    the large body of low-value lexical noise at eh=0.25 / conf=0.20-0.30.

    Returns a (has_signal, reason) tuple. When has_signal is False, the reason
    explains why the group was rejected.
    """
    # 18.4.1: structural support from any document.
    if group.best_title_support > 0:
        return True, "title_support"
    if group.best_possessive_support > 0:
        return True, "possessive_support"
    if group.best_location_support > 0:
        return True, "location_support"
    if group.best_linked_field > 0:
        return True, "linked_field_support"
    if group.best_linked_definition > 0:
        return True, "linked_definition_support"
    if group.best_linked_seed > 0:
        return True, "linked_seed_support"

    # 18.4.2: strong deterministic plausibility.
    if group.best_entityhood >= 0.55:
        return True, "strong_entityhood"
    if group.best_confidence >= 0.40:
        return True, "strong_confidence"

    # 18.4.3: mixed category evidence across documents.
    non_unresolved = group.winning_categories - {"unresolved"}
    if "unresolved" in group.winning_categories and non_unresolved:
        return True, f"mixed_categories"

    # 18.4.4: overlap fragment with independent uncovered support.
    if group.has_independent_overlap_fragment:
        return True, "independent_overlap_fragment"

    # 18.4.5: compound participation - the key appears as a word-token
    # inside a multi-token compound candidate (surviving or suppressed).
    # This is weak but positive evidence of name-ness: "onitsuka" in
    # "onitsuka natsumi" is more likely a surname than noise.
    if group.has_compound_participation:
        return True, "compound_participation"

    # 18.4.6: total occurrence count across all documents (including
    # non-suppressed). Keys that recur heavily without structural support
    # are still worth an LLM opinion when no other signal is available.
    if group.cross_doc_total_occurrences >= 5:
        return True, "high_total_occurrences"

    return False, "no_positive_rescue_signal"


def build_rescue_task_packets(
    bundle: ManuscriptReviewBundle,
    document_texts: dict[str, str],
) -> tuple[list[LLMTaskPacket], list[LLMTaskSelectionDiagnostic]]:
    """Build suppression rescue LLM task packets from a manuscript bundle.

    Records sharing the same normalized_key are merged into one packet
    so the LLM sees evidence from all documents in a single call.

    Args:
        bundle: Manuscript review bundle containing entity records.
        document_texts: Map of document path to raw text content, used
            to build evidence windows around rescue candidate anchors.

    Returns:
        Task packets for rescue candidates and selection diagnostics
        for all suppressed records evaluated.
    """
    diagnostics: list[LLMTaskSelectionDiagnostic] = []

    absorbed_keys: frozenset[str] = frozenset(
        source_key
        for entity in bundle.canonical_entities
        for source_key in entity.source_keys
        if source_key != entity.canonical_key
    )

    # Multi-token keys that survived extraction as promoted or review_only.
    # Used by the widened lineage check to reject components whose every
    # occurrence is explained by a surviving compound.
    surviving_compound_keys: frozenset[str] = frozenset(
        record.identity.normalized_key
        for record in bundle.entity_records
        if record.current_state.bucket in (DocumentEntityBucket.PROMOTED, DocumentEntityBucket.REVIEW_ONLY)
        and " " in record.identity.normalized_key
    )

    # Cross-document support index: for each normalized key, aggregate the
    # best support signals across ALL documents (including documents where
    # the key is promoted or review_only, not just suppressed). A key like
    # "firth" may be suppressed with eh=0.25 in three documents but
    # review_only with eh=0.80 and possessive support in four others. The
    # non-suppressed records carry the positive evidence that the key is a
    # real entity worth rescuing in its weak documents.
    _cross_doc_support: dict[str, dict[str, float | int | bool | set[str]]] = {}
    for record in bundle.entity_records:
        key = record.identity.normalized_key
        if key not in _cross_doc_support:
            _cross_doc_support[key] = {
                "best_entityhood": 0.0,
                "best_confidence": 0.0,
                "best_title": 0,
                "best_poss": 0,
                "best_loc": 0,
                "best_linked_field": 0,
                "best_linked_def": 0,
                "best_linked_seed": 0,
                "categories": set(),
                "has_overlap_uncovered": False,
                "has_compound_participation": False,
                "total_occurrences": 0,
            }
        s = _cross_doc_support[key]
        sp = record.support_profile
        s["best_entityhood"] = max(s["best_entityhood"], record.classification_trace.entityhood.score or 0.0)
        s["best_confidence"] = max(s["best_confidence"], record.promotion_trace.confidence_score or 0.0)
        s["best_title"] = max(s["best_title"], sp.title_support_count)
        s["best_poss"] = max(s["best_poss"], sp.possessive_support_count)
        s["best_loc"] = max(s["best_loc"], sp.location_support_count)
        s["best_linked_field"] = max(s["best_linked_field"], sp.linked_field_count)
        s["best_linked_def"] = max(s["best_linked_def"], sp.linked_definition_count)
        s["best_linked_seed"] = max(s["best_linked_seed"], sp.linked_seed_count)
        s["total_occurrences"] += record.source_evidence.occurrence_count
        s["categories"].add(record.current_state.winning_category.value)
        if (
            record.promotion_trace.suppression_reason == SuppressReason.COMPONENT_OVERLAP_NOISE
            and record.lineage_profile.uncovered_anchor_count > 0
        ):
            s["has_overlap_uncovered"] = True

    # A key that appears as a word-token inside any multi-token key
    # (surviving or suppressed) shows compound participation - evidence
    # that it co-occurs with other capitalized words as part of a name.
    all_compound_keys = {
        rec.identity.normalized_key
        for rec in bundle.entity_records
        if " " in rec.identity.normalized_key
    }
    compound_component_tokens: frozenset[str] = frozenset(
        token
        for ck in all_compound_keys
        for token in ck.split()
    )
    for key in _cross_doc_support:
        if key in compound_component_tokens:
            _cross_doc_support[key]["has_compound_participation"] = True

    # First pass: filter and group selected records by normalized_key.
    groups: dict[str, _RescueGroup] = {}

    for record in bundle.entity_records:
        selected, reason = _rescue_candidate_selected(
            record, absorbed_keys, surviving_compound_keys,
        )
        rescue_evidence: list[LLMTaskEvidenceItem] = []

        if selected:
            raw_text = document_texts.get(record.identity.document_anchor.path, "")
            if not raw_text:
                selected = False
                reason = "missing_document_text"
            else:
                rescue_evidence = _build_rescue_evidence(record, raw_text)
                if not rescue_evidence:
                    selected = False
                    reason = "no_rescue_evidence_windows"

        diagnostics.append(
            LLMTaskSelectionDiagnostic(
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="suppressed_entity_record",
                source_object_id=record.identity.normalized_key,
                document_path=record.identity.document_anchor.path,
                task_family=LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE,
                selected=selected,
                reason=reason,
                evidence_counts={
                    "occurrence_count": record.source_evidence.occurrence_count,
                    "scene_count": record.promotion_trace.scene_count,
                    "anchor_count": len(record.source_evidence.anchors),
                    "quote_only": int(record.discourse_profile.quote_only),
                    "address_like_count": record.discourse_profile.address_like_count,
                    "one_token_utterance_count": record.discourse_profile.one_token_utterance_count,
                    "uncovered_anchor_count": record.lineage_profile.uncovered_anchor_count,
                    "fully_covered_by_longer_compound": int(
                        record.lineage_profile.fully_covered_by_longer_compound
                    ),
                },
            )
        )
        if not selected:
            continue

        key = record.identity.normalized_key
        if key not in groups:
            # Seed group-level signals from the cross-document index so that
            # non-suppressed records' strength contributes to the gate.
            xdoc = _cross_doc_support.get(key, {})
            groups[key] = _RescueGroup(
                normalized_key=key,
                records=[],
                evidence=[],
                document_paths=[],
                surface_forms=set(),
                total_occurrences=0,
                total_scenes=0,
                suppression_reasons=set(),
                winning_categories=xdoc.get("categories", set()),
                best_confidence=xdoc.get("best_confidence", 0.0),
                best_entityhood=xdoc.get("best_entityhood", 0.0),
                best_title_support=xdoc.get("best_title", 0),
                best_possessive_support=xdoc.get("best_poss", 0),
                best_location_support=xdoc.get("best_loc", 0),
                best_linked_field=xdoc.get("best_linked_field", 0),
                best_linked_definition=xdoc.get("best_linked_def", 0),
                best_linked_seed=xdoc.get("best_linked_seed", 0),
                max_uncovered_anchors=0,
                has_independent_overlap_fragment=xdoc.get("has_overlap_uncovered", False),
                has_compound_participation=xdoc.get("has_compound_participation", False),
                cross_doc_total_occurrences=xdoc.get("total_occurrences", 0),
            )
        group = groups[key]
        group.records.append(record)
        group.evidence.extend(rescue_evidence)
        if record.identity.document_anchor.path not in group.document_paths:
            group.document_paths.append(record.identity.document_anchor.path)
        group.surface_forms.update(record.identity.surface_forms)
        group.total_occurrences += record.source_evidence.occurrence_count
        group.total_scenes += record.promotion_trace.scene_count
        if record.promotion_trace.suppression_reason is not None:
            group.suppression_reasons.add(record.promotion_trace.suppression_reason.value)
        group.max_uncovered_anchors = max(group.max_uncovered_anchors, record.lineage_profile.uncovered_anchor_count)

    # Second pass: apply group-level positive-signal gate, then build
    # one packet per surviving group. Groups without any positive
    # deterministic signal are rejected to avoid spending LLM budget
    # on low-value lexical noise.
    packets: list[LLMTaskPacket] = []
    for group in groups.values():
        has_signal, signal_reason = _rescue_group_has_positive_signal(group)
        if not has_signal:
            diagnostics.append(
                LLMTaskSelectionDiagnostic(
                    source_bundle_kind="manuscript_review_bundle",
                    source_object_kind="rescue_group",
                    source_object_id=group.normalized_key,
                    document_path=group.document_paths[0] if group.document_paths else "",
                    task_family=LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE,
                    selected=False,
                    reason=signal_reason,
                    evidence_counts={
                        "total_occurrences": group.total_occurrences,
                        "total_scenes": group.total_scenes,
                        "best_entityhood": group.best_entityhood,
                        "best_confidence": group.best_confidence,
                        "best_title_support": group.best_title_support,
                        "best_possessive_support": group.best_possessive_support,
                        "best_location_support": group.best_location_support,
                        "winning_categories": sorted(group.winning_categories),
                    },
                )
            )
            continue
        capped_evidence = group.evidence[:_RESCUE_EVIDENCE_LIMIT]
        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE.value,
                    group.normalized_key,
                ),
                task_family=LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE,
                schema_id=_SCHEMA_ID,
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="suppressed_entity_record",
                source_object_id=group.normalized_key,
                source_document_paths=group.document_paths,
                document_type=DocumentType.MANUSCRIPT,
                document_status=DocumentStatus.PRIMARY_CANON,
                source_authority="manuscript_corpus",
                source_authority_weight=1.0,
                task_goal=(
                    "Determine whether this suppressed entity mention is a genuine "
                    "recurring entity that was incorrectly filtered by deterministic rules."
                ),
                task_constraints=[
                    "Use only the surrounding manuscript context provided in evidence.",
                    (
                        "A genuine entity is a named character, place, group, ship, "
                        "title-as-name, or concept that recurs meaningfully in the narrative."
                    ),
                    "Generic English words that happen to be capitalized are not entities.",
                    "If a title or rank consistently refers to one character, rescue it.",
                ],
                evidence_payload=capped_evidence,
                selection_reason="rescue_candidate",
                payload={
                    "normalized_key": group.normalized_key,
                    "surface_forms": sorted(group.surface_forms),
                    "occurrence_count": group.total_occurrences,
                    "scene_count": group.total_scenes,
                    "suppression_reasons": sorted(group.suppression_reasons),
                    "winning_categories": sorted(group.winning_categories),
                    "confidence_score": group.best_confidence,
                    "entityhood_score": group.best_entityhood,
                },
            )
        )

    return packets, diagnostics
