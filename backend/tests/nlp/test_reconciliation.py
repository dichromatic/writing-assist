"""
Tests for backend/nlp/reconciliation/*.

These tests lock in the first corpus-refinement stage: stable per-document
entity snapshots plus conservative exact-key cross-document reconciliation.
"""

from pathlib import Path

from backend.nlp.classification import classify_clusters
from backend.nlp.lexicon.bootstrap import bootstrap
from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.promotion.attribution import attribute_dialogue
from backend.nlp.promotion.promotion import promote
from backend.nlp.reconciliation.corpus_entities import reconcile_document_entities
from backend.nlp.reconciliation.document_entities import summarize_document_entities
from backend.nlp.types import (
    CategoryEvidenceTrace,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityClassificationTrace,
    DocumentEntityCurrentState,
    DocumentEntityDiscourseProfile,
    DocumentEntityIdentity,
    DocumentEntityLineageProfile,
    DocumentEntityPromotionTrace,
    DocumentEntityRecord,
    DocumentEntitySourceEvidence,
    DocumentEntitySupportProfile,
    EntityhoodTrace,
    EvidenceWindow,
    LexiconCategory,
    SpanAnchor,
    SuppressReason,
)


def run_document_entities(text: str, path: str) -> list[DocumentEntityRecord]:
    """Run the existing document pipeline and return reconciliation records."""
    doc = parse(path, text)
    pre = preprocess(doc)
    result = bootstrap(doc)
    attribution_records = attribute_dialogue(pre, result.clusters)
    promotion_result = promote(pre, result.clusters, result.lexicon, attribution_records)
    return summarize_document_entities(
        pre,
        result.clusters,
        promotion_result.bundle,
        promotion_result.scores,
        promotion_result.classifications,
        attribution_records,
    )


def run_document_entity_inputs(text: str, path: str):
    """Return full summarize inputs and outputs for trace-level assertions."""
    doc = parse(path, text)
    pre = preprocess(doc)
    result = bootstrap(doc)
    attribution_records = attribute_dialogue(pre, result.clusters)
    promotion_result = promote(pre, result.clusters, result.lexicon, attribution_records)
    records = summarize_document_entities(
        pre,
        result.clusters,
        promotion_result.bundle,
        promotion_result.scores,
        promotion_result.classifications,
        attribution_records,
    )
    return result, promotion_result, records


def make_record(
    path: str,
    normalized_key: str,
    category: LexiconCategory,
    *,
    resolved: bool = True,
    confidence_score: float = 0.6,
    bucket: DocumentEntityBucket = DocumentEntityBucket.REVIEW_ONLY,
    suppression_reason: SuppressReason | None = None,
) -> DocumentEntityRecord:
    """Build a minimal document entity record for reconciliation tests."""
    anchor = SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(normalized_key))
    return DocumentEntityRecord(
        identity=DocumentEntityIdentity(
            record_id=f"test-{path}-{normalized_key}",
            document_anchor=DocumentAnchor(path=path),
            normalized_key=normalized_key,
            surface_forms=[normalized_key.title()],
        ),
        current_state=DocumentEntityCurrentState(
            winning_category=category,
            resolved=resolved,
            bucket=bucket,
        ),
        source_evidence=DocumentEntitySourceEvidence(
            occurrence_count=2,
            anchors=[anchor],
            evidence_windows=[
                EvidenceWindow(
                    entity_key=normalized_key,
                    anchor=anchor,
                    context_before="",
                    context_after="",
                    is_first_introduction=True,
                    has_attribution=False,
                    speaker=None,
                )
            ],
        ),
        classification_trace=DocumentEntityClassificationTrace(
            winning_score=0.6,
            runner_up_category=None,
            runner_up_score=0.0,
            evidence_by_category={
                category: CategoryEvidenceTrace(
                    category=category,
                    score=0.6,
                    reasons=[],
                    vetoes=[],
                )
            },
            entityhood=EntityhoodTrace(
                score=0.6,
                accepted=True,
                reasons=[],
                weaknesses=[],
            ),
        ),
        promotion_trace=DocumentEntityPromotionTrace(
            confidence_score=confidence_score,
            suppression_reason=suppression_reason,
            bucket_detail="",
            rule_tier=2,
            scene_count=1,
            attribution_count=0,
            possessive_count=0,
            tfidf_score=0.0,
        ),
        discourse_profile=DocumentEntityDiscourseProfile(
            in_quote_count=0,
            non_quote_count=2,
            quote_only=False,
            sentence_initial_count=0,
            sentence_initial_only=False,
            address_like_count=0,
            attributed_speaker_nearby_count=0,
            one_token_utterance_count=0,
        ),
        support_profile=DocumentEntitySupportProfile(
            title_support_count=0,
            possessive_support_count=0,
            location_support_count=0,
            linked_field_count=0,
            linked_definition_count=0,
            linked_seed_count=0,
        ),
        lineage_profile=DocumentEntityLineageProfile(
            compound_part_count=len(normalized_key.split()),
            fully_covered_by_longer_compound=False,
            candidate_parent_keys=[],
            covered_anchor_count=0,
            uncovered_anchor_count=1,
            appears_as_compound_component=False,
            appears_as_compound_surface=(" " in normalized_key),
        ),
    )


class TestDocumentEntitySummaries:
    def test_summary_preserves_document_bucket_and_category(self):
        # The reconciliation layer must preserve what the document pipeline
        # already decided, rather than re-inferring promotion status later.
        text = (
            "Captain Aldous arrived.\n\n"
            "---\n\n"
            "In Sidhe, bells rang. Sidhe slept."
        )
        records = run_document_entities(text, "doc-a.md")
        by_key = {record.identity.normalized_key: record for record in records}

        assert by_key["aldous"].current_state.bucket == DocumentEntityBucket.PROMOTED
        assert by_key["aldous"].current_state.winning_category == LexiconCategory.CHARACTER

        assert by_key["sidhe"].current_state.bucket == DocumentEntityBucket.REVIEW_ONLY
        assert by_key["sidhe"].current_state.winning_category == LexiconCategory.PLACE

    def test_suppressed_overlap_attaches_to_stronger_local_entity_record(self):
        # Suppressed overlap fragments should not disappear entirely before the
        # later semantic pass. When they sit beneath a stronger local entity,
        # they should travel with that entity as retained secondary evidence.
        path = "examples/4. Tairngire.md"
        records = run_document_entities(Path(path).read_text(encoding="utf-8"), path)
        by_key = {record.identity.normalized_key: record for record in records}

        assert "lantern festival" in by_key
        attached_keys = {
            evidence.normalized_key
            for evidence in by_key["lantern festival"].source_evidence.suppressed_related_evidence
        }
        assert "lantern" in attached_keys
        assert "festival" in attached_keys

    def test_summary_preserves_full_classification_trace_from_classifier(self):
        # The record-local classification trace should preserve all category
        # and entityhood evidence from classification so later stages do not
        # need to recover dropped decision details.
        result, promotion_result, records = run_document_entity_inputs(
            "Captain Aldous arrived in Sidhe. Sidhe bells rang.",
            "doc-trace.md",
        )
        by_key = {record.identity.normalized_key: record for record in records}

        for cluster in result.clusters:
            key = cluster.normalized_key
            source = promotion_result.classifications[key]
            trace = by_key[key].classification_trace

            assert trace.winning_score == source.winning_score
            assert trace.runner_up_category == source.runner_up_category
            assert trace.runner_up_score == source.runner_up_score
            assert set(trace.evidence_by_category) == set(source.evidence_by_category)
            for category, evidence in source.evidence_by_category.items():
                mapped = trace.evidence_by_category[category]
                assert mapped.category == evidence.category
                assert mapped.score == evidence.score
                assert mapped.reasons == evidence.reasons
                assert mapped.vetoes == evidence.vetoes
            assert trace.entityhood.score == source.entityhood.score
            assert trace.entityhood.accepted == source.entityhood.accepted
            assert trace.entityhood.reasons == source.entityhood.reasons
            assert trace.entityhood.weaknesses == source.entityhood.weaknesses

    def test_summary_preserves_promotion_trace_signals_and_bucket_details(self):
        # Promotion trace fields capture suppression and routing evidence that
        # is not reconstructible from the final bucket alone.
        result, promotion_result, records = run_document_entity_inputs(
            Path("examples/4. Tairngire.md").read_text(encoding="utf-8"),
            "examples/4. Tairngire.md",
        )
        by_key = {record.identity.normalized_key: record for record in records}
        suppressed_by_key = {
            candidate.cluster.normalized_key: candidate
            for candidate in promotion_result.bundle.suppressed
        }

        for cluster in result.clusters:
            key = cluster.normalized_key
            signals, score = promotion_result.scores[key]
            trace = by_key[key].promotion_trace

            assert trace.confidence_score == score
            assert trace.rule_tier == signals.rule_tier
            assert trace.scene_count == signals.scene_count
            assert trace.attribution_count == signals.attribution_count
            assert trace.possessive_count == signals.possessive_count
            assert trace.tfidf_score == signals.tfidf_score
            if key in suppressed_by_key:
                assert trace.suppression_reason == suppressed_by_key[key].reason
                assert trace.bucket_detail == suppressed_by_key[key].detail

    def test_summary_derives_discourse_profile_for_quote_and_address_usage(self):
        # Quote membership, direct address, and sentence-initial usage are
        # separate discourse signals and should be preserved distinctly.
        records = run_document_entities(
            '"Aldous," Kohaku said. Aldous waited.',
            "doc-discourse.md",
        )
        aldous = next(record for record in records if record.identity.normalized_key == "aldous")

        assert aldous.discourse_profile.in_quote_count == 1
        assert aldous.discourse_profile.non_quote_count == 1
        assert aldous.discourse_profile.quote_only is False
        assert aldous.discourse_profile.address_like_count == 1
        assert aldous.discourse_profile.sentence_initial_count >= 1
        assert aldous.discourse_profile.attributed_speaker_nearby_count == 1

    def test_summary_tracks_lineage_coverage_counts_for_component_vs_independent_use(self):
        # Coverage counts should distinguish component-only appearances inside
        # longer compounds from independent standalone support.
        records = run_document_entities(
            Path("examples/4. Tairngire.md").read_text(encoding="utf-8"),
            "examples/4. Tairngire.md",
        )
        by_key = {record.identity.normalized_key: record for record in records}
        lantern = by_key["lantern"]

        assert lantern.lineage_profile.covered_anchor_count > 0
        assert "lantern festival" in lantern.lineage_profile.candidate_parent_keys
        assert (
            lantern.lineage_profile.covered_anchor_count
            + lantern.lineage_profile.uncovered_anchor_count
            == lantern.source_evidence.occurrence_count
        )


class TestCorpusReconciliation:
    def test_exact_key_records_merge_into_one_corpus_entity(self):
        # The first corpus phase should merge exact-key records across
        # documents into one canonical entity rather than treating each
        # document-local record as independent forever.
        records = [
            make_record("a.md", "aldous", LexiconCategory.CHARACTER, confidence_score=0.70),
            make_record("b.md", "aldous", LexiconCategory.CHARACTER, confidence_score=0.55),
        ]

        result = reconcile_document_entities(records)

        assert len(result.canonical_entities) == 1
        entity = result.canonical_entities[0]
        assert entity.canonical_key == "aldous"
        assert entity.source_keys == ["aldous"]
        assert entity.dominant_category == LexiconCategory.CHARACTER
        assert entity.supporting_document_paths == ["a.md", "b.md"]
        assert entity.review_required is False

    def test_conflicting_resolved_categories_flag_review(self):
        # Same-key entities with incompatible resolved categories must remain
        # visible for review rather than being silently collapsed.
        records = [
            make_record("a.md", "meridian", LexiconCategory.GROUP, confidence_score=0.65),
            make_record("b.md", "meridian", LexiconCategory.PLACE, confidence_score=0.62),
        ]

        result = reconcile_document_entities(records)

        entity = result.canonical_entities[0]
        assert entity.canonical_key == "meridian"
        assert entity.review_required is True
        assert set(entity.conflicting_categories) == {
            LexiconCategory.GROUP,
            LexiconCategory.PLACE,
        }

    def test_resolved_category_beats_unresolved_when_key_matches(self):
        # An unresolved document-local record should not erase a resolved
        # category seen elsewhere for the same exact key.
        records = [
            make_record("a.md", "tairngire", LexiconCategory.PLACE, confidence_score=0.45),
            make_record(
                "b.md",
                "tairngire",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                confidence_score=0.30,
            ),
        ]

        result = reconcile_document_entities(records)

        entity = result.canonical_entities[0]
        assert entity.dominant_category == LexiconCategory.PLACE
        assert entity.review_required is False

    def test_character_compound_absorbs_single_token_alias_components(self):
        # A resolved character full name should become the canonical corpus
        # entity when both single-token parts also support the same person.
        records = [
            make_record("a.md", "tsushima yoshiko", LexiconCategory.CHARACTER, confidence_score=0.82),
            make_record("a.md", "tsushima", LexiconCategory.CHARACTER, confidence_score=0.70),
            make_record("a.md", "yoshiko", LexiconCategory.CHARACTER, confidence_score=0.75),
            make_record("b.md", "yoshiko", LexiconCategory.CHARACTER, confidence_score=0.78),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["tsushima yoshiko"]
        entity = result.canonical_entities[0]
        assert entity.source_keys == ["tsushima", "tsushima yoshiko", "yoshiko"]
        assert entity.canonical_surface_forms == ["Tsushima Yoshiko"]
        assert entity.absorbed_surface_forms == ["Tsushima", "Yoshiko"]
        assert entity.supporting_document_paths == ["a.md", "b.md"]
        assert len(entity.member_records) == 4
        assert "character compound merged with its single-token alias components" in entity.reasons

    def test_ambiguous_component_does_not_merge_multiple_character_compounds(self):
        # If a single-token component could belong to more than one character
        # compound, reconciliation should keep the exact entities separate
        # rather than guessing which canonical person owns the shared token.
        records = [
            make_record("a.md", "tsushima yoshiko", LexiconCategory.CHARACTER, confidence_score=0.82),
            make_record("a.md", "takami yoshiko", LexiconCategory.CHARACTER, confidence_score=0.80),
            make_record("a.md", "tsushima", LexiconCategory.CHARACTER, confidence_score=0.70),
            make_record("a.md", "takami", LexiconCategory.CHARACTER, confidence_score=0.72),
            make_record("a.md", "yoshiko", LexiconCategory.CHARACTER, confidence_score=0.78),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == [
            "takami",
            "takami yoshiko",
            "tsushima",
            "tsushima yoshiko",
            "yoshiko",
        ]

    def test_title_like_character_compound_prefers_personal_key_as_canonical(self):
        # A title-led or role-led person phrase is useful alias evidence, but
        # the shorter personal key is usually the better canonical anchor.
        records = [
            make_record("a.md", "elder earlean", LexiconCategory.CHARACTER, confidence_score=0.95),
            make_record(
                "a.md",
                "elder",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                confidence_score=0.50,
            ),
            make_record("a.md", "earlean", LexiconCategory.CHARACTER, confidence_score=0.92),
            make_record("b.md", "earlean", LexiconCategory.CHARACTER, confidence_score=0.88),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["earlean"]
        entity = result.canonical_entities[0]
        assert entity.source_keys == ["earlean", "elder", "elder earlean"]
        assert "titled or role-led compound deferred to stronger personal key" in entity.reasons

    def test_suppressed_full_name_anchor_can_still_merge_character_aliases(self):
        # A full-name surface observed only once is still valuable alias
        # evidence. If the compound itself is suppressed but both components
        # are strong character entities, corpus reconciliation should still be
        # able to use that observed full form as the canonical identity.
        records = [
            make_record(
                "a.md",
                "tsushima yoshiko",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                confidence_score=0.20,
                bucket=DocumentEntityBucket.SUPPRESSED,
            ),
            make_record("a.md", "tsushima", LexiconCategory.CHARACTER, confidence_score=0.70),
            make_record("a.md", "yoshiko", LexiconCategory.CHARACTER, confidence_score=0.75),
            make_record("b.md", "yoshiko", LexiconCategory.CHARACTER, confidence_score=0.78),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["tsushima yoshiko"]
        entity = result.canonical_entities[0]
        assert entity.source_keys == ["tsushima", "tsushima yoshiko", "yoshiko"]
        assert len(entity.member_records) == 4
        assert "sparse observed character compound merged with its single-token alias components" in entity.reasons

    def test_review_conflicted_character_component_can_still_support_full_name_merge(self):
        # A component key can be noisy elsewhere in the corpus while still
        # being character-dominant enough to support a real full-name merge.
        # The merged canonical should stay reviewable rather than refusing to
        # form at all.
        records = [
            make_record(
                "a.md",
                "tsushima yoshiko",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                confidence_score=0.20,
                bucket=DocumentEntityBucket.SUPPRESSED,
            ),
            make_record("a.md", "tsushima", LexiconCategory.CHARACTER, confidence_score=0.70),
            make_record("a.md", "yoshiko", LexiconCategory.CHARACTER, confidence_score=0.75),
            make_record("b.md", "yoshiko", LexiconCategory.PLACE, confidence_score=0.40),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["tsushima yoshiko"]
        entity = result.canonical_entities[0]
        assert entity.review_required is True
        assert LexiconCategory.PLACE in entity.conflicting_categories

    def test_suppressed_records_are_excluded_from_canonical_entities_by_default(self):
        # Canonical corpus entities should default to promoted/review evidence.
        # Suppressed noise is still useful for diagnostics later, but it should
        # not pollute the main exact-key entity set by default.
        records = [
            make_record("a.md", "aldous", LexiconCategory.CHARACTER),
            make_record(
                "b.md",
                "hey",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                bucket=DocumentEntityBucket.SUPPRESSED,
                confidence_score=0.20,
            ),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["aldous"]

    def test_rescued_suppressed_keys_are_included_without_global_suppressed_toggle(self):
        # Rescue verdicts should allow specific suppressed keys through the
        # reconciliation gate without enabling all suppressed records.
        records = [
            make_record("a.md", "aldous", LexiconCategory.CHARACTER),
            make_record(
                "b.md",
                "firth",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                bucket=DocumentEntityBucket.SUPPRESSED,
                confidence_score=0.20,
            ),
            make_record(
                "c.md",
                "hey",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                bucket=DocumentEntityBucket.SUPPRESSED,
                confidence_score=0.20,
            ),
        ]

        result = reconcile_document_entities(records, rescued_keys=frozenset({"firth"}))

        assert [entity.canonical_key for entity in result.canonical_entities] == ["aldous", "firth"]

    def test_generic_leading_character_compounds_defer_to_trailing_personal_key(self):
        # Generic leading modifiers such as "old" and "man" are useful alias
        # surfaces, but when the trailing personal key is already stronger they
        # should not become separate canonicals.
        records = [
            make_record("a.md", "hiroshi", LexiconCategory.CHARACTER, confidence_score=0.70),
            make_record("a.md", "man hiroshi", LexiconCategory.CHARACTER, confidence_score=0.40),
            make_record("a.md", "old man hiroshi", LexiconCategory.CHARACTER, confidence_score=0.40),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["hiroshi"]
        entity = result.canonical_entities[0]
        assert entity.source_keys == ["hiroshi", "man hiroshi", "old man hiroshi"]
        assert "generic-leading character compounds deferred to stronger personal key" in entity.reasons

    def test_resolved_group_compound_absorbs_shorter_head_alias(self):
        # Institutional head nouns such as "institute" are often reused as a
        # shorter reference to a fully named group. When the shorter head only
        # appears inside the same document set, the full compound should be the
        # canonical corpus key.
        records = [
            make_record("a.md", "norre institute", LexiconCategory.GROUP, confidence_score=0.60),
            make_record("b.md", "norre institute", LexiconCategory.GROUP, confidence_score=0.65),
            make_record("a.md", "institute", LexiconCategory.GROUP, confidence_score=0.40),
            make_record("b.md", "institute", LexiconCategory.GROUP, confidence_score=0.35),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["norre institute"]
        entity = result.canonical_entities[0]
        assert entity.source_keys == ["institute", "norre institute"]
        assert "resolved non-character compound absorbed its shorter head alias" in entity.reasons

    def test_shared_non_character_head_does_not_merge_multiple_compounds(self):
        # A generic head such as "council" can belong to several compounds.
        # Reconciliation should keep the shorter key visible when it does not
        # uniquely identify one stronger compound.
        records = [
            make_record("a.md", "council", LexiconCategory.GROUP, confidence_score=0.40),
            make_record("a.md", "magical council", LexiconCategory.GROUP, confidence_score=0.45),
            make_record("a.md", "recovery council", LexiconCategory.GROUP, confidence_score=0.45),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == [
            "council",
            "magical council",
            "recovery council",
        ]

    def test_resolved_place_compound_absorbs_shorter_modifier_alias(self):
        # A modifier-only place key such as "radiant" is often just a shorter
        # surface for one stronger named place compound in the same document
        # set. The resolved compound should become the canonical corpus key.
        records = [
            make_record("a.md", "radiant estuary", LexiconCategory.PLACE, confidence_score=0.55),
            make_record("b.md", "radiant estuary", LexiconCategory.PLACE, confidence_score=0.60),
            make_record("a.md", "radiant", LexiconCategory.PLACE, confidence_score=0.30),
            make_record("b.md", "radiant", LexiconCategory.PLACE, confidence_score=0.28),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["radiant estuary"]
        entity = result.canonical_entities[0]
        assert entity.source_keys == ["radiant", "radiant estuary"]
        assert "resolved non-character compound absorbed its shorter modifier alias" in entity.reasons

    def test_shared_non_character_modifier_does_not_merge_multiple_compounds(self):
        # A modifier such as "east" can prefix more than one place compound.
        # Reconciliation should keep the shorter modifier key visible until the
        # corpus provides a unique stronger target.
        records = [
            make_record("a.md", "east", LexiconCategory.PLACE, confidence_score=0.25),
            make_record("a.md", "east lagoon", LexiconCategory.PLACE, confidence_score=0.35),
            make_record("a.md", "east harbor", LexiconCategory.PLACE, confidence_score=0.35),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == [
            "east",
            "east harbor",
            "east lagoon",
        ]

    def test_resolved_longer_place_compound_absorbs_shorter_contained_alias(self):
        # A longer resolved place name should absorb a shorter contained alias
        # when the corpus shows the shorter phrase only as the same place.
        records = [
            make_record(
                "a.md",
                "amerhinn remembrance gardens",
                LexiconCategory.PLACE,
                confidence_score=0.30,
            ),
            make_record(
                "a.md",
                "remembrance gardens",
                LexiconCategory.PLACE,
                confidence_score=0.20,
            ),
            make_record(
                "a.md",
                "remembrance",
                LexiconCategory.PLACE,
                confidence_score=0.20,
            ),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == [
            "amerhinn remembrance gardens",
        ]
        entity = result.canonical_entities[0]
        assert entity.source_keys == [
            "amerhinn remembrance gardens",
            "remembrance",
            "remembrance gardens",
        ]
        assert "resolved non-character compound absorbed its shorter contained alias" in entity.reasons

    def test_shared_contained_alias_does_not_merge_multiple_longer_compounds(self):
        # A shorter multi-token phrase such as "remembrance gardens" can still
        # belong to more than one longer place compound. Reconciliation should
        # keep the shorter alias visible until ownership is unique.
        records = [
            make_record("a.md", "remembrance gardens", LexiconCategory.PLACE, confidence_score=0.20),
            make_record(
                "a.md",
                "amerhinn remembrance gardens",
                LexiconCategory.PLACE,
                confidence_score=0.30,
            ),
            make_record(
                "a.md",
                "uchiura remembrance gardens",
                LexiconCategory.PLACE,
                confidence_score=0.30,
            ),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == [
            "amerhinn remembrance gardens",
            "remembrance gardens",
            "uchiura remembrance gardens",
        ]

    def test_longer_unresolved_compound_defers_to_resolved_non_character_anchor(self):
        # A weak longer unresolved phrase should disappear as its own canonical
        # when the corpus already has one stronger resolved non-character anchor
        # that owns the contained alias.
        records = [
            make_record("a.md", "east lagoon", LexiconCategory.PLACE, confidence_score=0.20),
            make_record(
                "a.md",
                "east lagoon villa",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                confidence_score=0.20,
            ),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == ["east lagoon"]
        entity = result.canonical_entities[0]
        assert entity.source_keys == ["east lagoon", "east lagoon villa"]
        assert "longer unresolved compound deferred to stronger resolved non-character anchor" in entity.reasons

    def test_longer_unresolved_compound_stays_separate_when_multiple_resolved_anchors_exist(self):
        # A longer unresolved phrase should stay visible when both its prefix
        # and suffix point to different resolved non-character anchors.
        records = [
            make_record("a.md", "east lagoon", LexiconCategory.PLACE, confidence_score=0.20),
            make_record("a.md", "lagoon villa", LexiconCategory.PLACE, confidence_score=0.20),
            make_record(
                "a.md",
                "east lagoon villa",
                LexiconCategory.UNRESOLVED,
                resolved=False,
                confidence_score=0.20,
            ),
        ]

        result = reconcile_document_entities(records)

        assert [entity.canonical_key for entity in result.canonical_entities] == [
            "east lagoon",
            "east lagoon villa",
            "lagoon villa",
        ]
