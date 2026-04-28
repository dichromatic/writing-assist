"""Tests for backend/nlp/semantic_review/*."""

from backend.nlp.reconciliation.document_entities import summarize_document_entities
from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictSource,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityRecord,
    EvidenceWindow,
    LexiconCategory,
    ReferenceCandidate,
    ReferenceCandidateType,
    ReferenceCluster,
    ReviewTaskKind,
    SpanAnchor,
)
from backend.nlp.lexicon.bootstrap import bootstrap
from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.promotion.attribution import attribute_dialogue
from backend.nlp.promotion.promotion import promote
from backend.nlp.reconciliation.corpus_entities import reconcile_document_entities
from backend.nlp.semantic_review import (
    build_character_summaries,
    build_reference_clusters,
    build_conflict_records,
    build_review_tasks,
    extract_reference_candidates,
    extract_title_role_candidates,
)


def _document_records(text: str, path: str = "doc.md"):
    """Run the current document pipeline and return preprocessing plus records."""
    doc = parse(path, text)
    pre = preprocess(doc)
    result = bootstrap(doc)
    attribution_records = attribute_dialogue(pre, result.clusters)
    bundle = promote(pre, result.clusters, result.lexicon, attribution_records)
    records = summarize_document_entities(pre, result.clusters, attribution_records, bundle)
    return pre, records


def _document_outputs(text: str, path: str = "doc.md"):
    """Run the current document pipeline and return preprocessing, records, and attributions."""
    doc = parse(path, text)
    pre = preprocess(doc)
    result = bootstrap(doc)
    attribution_records = attribute_dialogue(pre, result.clusters)
    bundle = promote(pre, result.clusters, result.lexicon, attribution_records)
    records = summarize_document_entities(pre, result.clusters, attribution_records, bundle)
    return pre, records, attribution_records


def _make_record(
    path: str,
    normalized_key: str,
    category: LexiconCategory,
    *,
    resolved: bool = True,
    confidence_score: float = 0.6,
    bucket: DocumentEntityBucket = DocumentEntityBucket.REVIEW_ONLY,
) -> DocumentEntityRecord:
    """Build a minimal document entity record for semantic-review tests."""
    return DocumentEntityRecord(
        document_anchor=DocumentAnchor(path=path),
        normalized_key=normalized_key,
        surface_forms=[normalized_key.title()],
        winning_category=category,
        resolved=resolved,
        entityhood_score=0.6,
        entityhood_accepted=True,
        confidence_score=confidence_score,
        bucket=bucket,
        bucket_detail="",
        occurrence_count=2,
        rule_tier=2,
        scene_count=1,
        attribution_count=0,
        has_title_support=False,
        has_possessive_support=False,
        anchors=[SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(normalized_key))],
        evidence_windows=[
            EvidenceWindow(
                entity_key=normalized_key,
                anchor=SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(normalized_key)),
                context_before="",
                context_after="",
                is_first_introduction=True,
                has_attribution=False,
                speaker=None,
            )
        ],
    )


class TestReferenceCandidates:
    def test_bound_and_lowercase_bare_titles_are_preserved(self):
        # Fiction often reuses both "Captain Aldous" and bare lowercase
        # mentions like "the captain". Losing either form would make later
        # semantic attachment review much weaker.
        pre, records = _document_records(
            "Captain Aldous arrived. The captain told Aldous to wait."
        )

        candidates = extract_title_role_candidates(pre, records, [])
        by_type_and_surface = {
            (candidate.reference_type.value, candidate.normalized): candidate
            for candidate in candidates
        }

        assert ("bound_title_role", "captain") in by_type_and_surface
        assert ("bare_title_role", "captain") in by_type_and_surface
        assert by_type_and_surface[("bound_title_role", "captain")].linked_entity_keys == ["aldous"]
        assert by_type_and_surface[("bare_title_role", "captain")].linked_entity_keys == ["aldous"]

    def test_repeated_bare_titles_group_into_one_reference_cluster(self):
        # Repeated bare titles in one document should become one grouped review
        # object so later semantic review sees ranked attachment evidence
        # instead of a flat mention list.
        pre, records = _document_records(
            "Captain Aldous arrived. The captain nodded. The captain waited."
        )

        candidates = extract_title_role_candidates(pre, records, [])
        clusters = build_reference_clusters(candidates)
        bare_captain = next(
            cluster for cluster in clusters
            if cluster.reference_type.value == "bare_title_role"
            and cluster.normalized == "captain"
        )

        assert bare_captain.occurrence_count == 2
        assert bare_captain.candidate_entity_scores == {"aldous": 2}

    def test_bare_title_cluster_does_not_inherit_all_bound_title_seeds(self):
        # Bound title mentions elsewhere in the document are useful fallback
        # evidence, but they should not be appended wholesale to every bare
        # title cluster. Otherwise one bare "captain" mention turns into an
        # unreadable candidate list of everyone who ever held that title.
        pre, records = _document_records(
            "Captain Aldous arrived. Captain Beatrix left. The captain waited."
        )

        candidates = extract_title_role_candidates(pre, records, [])
        clusters = build_reference_clusters(candidates)
        bare_captain = next(
            cluster for cluster in clusters
            if cluster.reference_type.value == "bare_title_role"
            and cluster.normalized == "captain"
        )

        assert bare_captain.candidate_entity_scores == {}

    def test_bare_title_without_linked_entity_still_emits_review_task(self):
        # A bare title with no deterministic attachment target is still useful
        # semantic evidence. It should survive into review rather than being
        # dropped because no nearby entity could be linked.
        pre, records = _document_records("The captain waited in silence.")

        candidates = extract_title_role_candidates(pre, records, [])
        tasks = build_review_tasks(build_reference_clusters(candidates), [])

        assert any(task.kind == ReviewTaskKind.TITLE_ROLE_ATTACHMENT for task in tasks)

    def test_bare_relation_without_linked_entity_emits_relation_review_task(self):
        # Kinship and relation nouns are also useful deferred references. They
        # should survive into review even when local deterministic attachment
        # cannot yet decide who they point to.
        pre, records = _document_records("The mother waited in silence.")

        candidates = extract_reference_candidates(pre, records, [])
        relation_candidates = [
            candidate for candidate in candidates
            if candidate.reference_type == ReferenceCandidateType.BARE_RELATION_ROLE
            and candidate.normalized == "mother"
        ]
        tasks = build_review_tasks(build_reference_clusters(candidates), [])

        assert relation_candidates
        assert any(task.kind == ReviewTaskKind.RELATION_ROLE_ATTACHMENT for task in tasks)

    def test_bound_and_bare_relations_are_preserved(self):
        # Relation nouns can appear both attached to a named character and as
        # later bare mentions. Both forms need to survive because later
        # semantic review resolves the stable relationship pattern, not the
        # extractor.
        doc = parse("doc.md", "Brother Connall arrived. The brother waited.")
        pre = preprocess(doc)
        records = [
            _make_record("doc.md", "connall", LexiconCategory.CHARACTER),
        ]
        records[0].anchors = [
            SpanAnchor(path="doc.md", span_ordinal=0, start_char=8, end_char=15),
            SpanAnchor(path="doc.md", span_ordinal=0, start_char=29, end_char=36),
        ]

        candidates = extract_reference_candidates(pre, records, [])
        by_type_and_surface = {
            (candidate.reference_type.value, candidate.normalized): candidate
            for candidate in candidates
        }

        assert ("bound_relation_role", "brother") in by_type_and_surface
        assert ("bare_relation_role", "brother") in by_type_and_surface
        assert by_type_and_surface[("bound_relation_role", "brother")].linked_entity_keys == ["connall"]
        assert by_type_and_surface[("bare_relation_role", "brother")].linked_entity_keys == ["connall"]

    def test_relation_candidates_collapse_overlapping_character_variants(self):
        # Relation prompts should not list stacked overlapping variants of the
        # same person as separate plausible targets. The local ranking step
        # should keep the strongest nearby character anchor and drop weaker
        # overlapping variants from the review prompt.
        doc = parse("doc.md", "The child saw old man Hiroshi.")
        pre = preprocess(doc)
        records = [
            _make_record("doc.md", "hiroshi", LexiconCategory.CHARACTER, confidence_score=0.80),
            _make_record("doc.md", "man hiroshi", LexiconCategory.CHARACTER, confidence_score=0.60),
            _make_record("doc.md", "old man hiroshi", LexiconCategory.CHARACTER, confidence_score=0.55),
        ]
        records[0].anchors = [SpanAnchor(path="doc.md", span_ordinal=0, start_char=22, end_char=29)]
        records[1].anchors = [SpanAnchor(path="doc.md", span_ordinal=0, start_char=18, end_char=29)]
        records[2].anchors = [SpanAnchor(path="doc.md", span_ordinal=0, start_char=14, end_char=29)]

        candidates = extract_reference_candidates(pre, records, [])
        child = next(
            candidate for candidate in candidates
            if candidate.reference_type == ReferenceCandidateType.BARE_RELATION_ROLE
            and candidate.normalized == "child"
        )

        assert child.linked_entity_keys == ["hiroshi"]

    def test_bare_title_inside_quote_vocative_is_marked_address_like(self):
        # Direct-address title uses are a different semantic problem than
        # descriptive mentions. The extractor should preserve that difference
        # so later review can treat "Captain, wait" as likely addressee
        # evidence rather than local self-description.
        pre, records, attribution_records = _document_outputs('"Captain, wait," Kohaku said.')

        candidates = extract_reference_candidates(pre, records, attribution_records)
        captain = next(
            candidate for candidate in candidates
            if candidate.reference_type == ReferenceCandidateType.BARE_TITLE_ROLE
            and candidate.normalized == "captain"
        )

        assert captain.address_like is True
        assert captain.quote_speaker_key == "kohaku"

    def test_address_like_reference_cluster_marks_review_prompt(self):
        # Once grouped, address-like references should stay visible in the
        # review prompt so later semantic review can recognize that the local
        # linked character may be the speaker rather than the target.
        pre, records, attribution_records = _document_outputs('"Captain, wait," Kohaku said.')

        candidates = extract_reference_candidates(pre, records, attribution_records)
        tasks = build_review_tasks(build_reference_clusters(candidates), [])
        captain_task = next(
            task for task in tasks
            if task.kind == ReviewTaskKind.TITLE_ROLE_ATTACHMENT
            and task.subject_key == "captain"
        )

        assert "address-like bare title 'captain'" in captain_task.prompt

    def test_address_like_reference_prompt_mentions_quote_speaker(self):
        # When a quote speaker is known, address-like review prompts should
        # surface that speaker explicitly. That keeps the later semantic pass
        # from treating the locally salient speaker as interchangeable with the
        # likely addressee.
        pre, records, attribution_records = _document_outputs('"Captain, wait," Kohaku said.')
        candidates = extract_reference_candidates(pre, records, attribution_records)
        tasks = build_review_tasks(build_reference_clusters(candidates), [])
        captain_task = next(
            task for task in tasks
            if task.kind == ReviewTaskKind.TITLE_ROLE_ATTACHMENT
            and task.subject_key == "captain"
        )

        assert "spoken by kohaku" in captain_task.prompt

    def test_address_like_prompt_does_not_offer_only_the_speaker_as_target(self):
        # When the only local candidate is the known speaker, the review
        # prompt should state the real question directly instead of pretending
        # that the speaker is the likely addressee.
        clusters = build_reference_clusters([
            ReferenceCandidate(
                document_anchor=DocumentAnchor(path="doc.md"),
                reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
                surface="Captain",
                normalized="captain",
                anchor=SpanAnchor(path="doc.md", span_ordinal=0, start_char=1, end_char=8),
                context_before="\"",
                context_after=", wait,\" Kohaku said.",
                in_quote=True,
                address_like=True,
                quote_speaker_key="kohaku",
                linked_entity_keys=["kohaku"],
            ),
        ])
        tasks = build_review_tasks(clusters, [])
        captain_task = next(
            task for task in tasks
            if task.kind == ReviewTaskKind.TITLE_ROLE_ATTACHMENT
            and task.subject_key == "captain"
        )

        assert "other than the speaker" in captain_task.prompt

    def test_address_like_prompt_uses_corpus_title_owners_as_fallback(self):
        # Address-like bare titles often only know the speaker locally. When
        # the corpus already has recurring title ownership evidence for other
        # characters, the review prompt should surface those owners instead of
        # stopping at a generic unresolved question.
        clusters = build_reference_clusters([
            ReferenceCandidate(
                document_anchor=DocumentAnchor(path="a.md"),
                reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
                surface="Captain",
                normalized="captain",
                anchor=SpanAnchor(path="a.md", span_ordinal=0, start_char=1, end_char=8),
                context_before="\"",
                context_after=", wait,\" Kohaku said.",
                in_quote=True,
                address_like=True,
                quote_speaker_key="kohaku",
                linked_entity_keys=["kohaku"],
            ),
            ReferenceCandidate(
                document_anchor=DocumentAnchor(path="b.md"),
                reference_type=ReferenceCandidateType.BOUND_TITLE_ROLE,
                surface="Captain Yō",
                normalized="captain",
                anchor=SpanAnchor(path="b.md", span_ordinal=0, start_char=0, end_char=11),
                context_before="",
                context_after=" arrived.",
                in_quote=False,
                address_like=False,
                quote_speaker_key=None,
                linked_entity_keys=["yō"],
            ),
        ])
        tasks = build_review_tasks(
            clusters,
            [],
            [
                CharacterSemanticSummary(
                    canonical_key="watanabe yō",
                    alias_keys=["watanabe", "yō"],
                    supporting_document_paths=["b.md"],
                    attached_title_counts={"captain": 5},
                    ambiguous_title_counts={},
                    attached_relation_counts={},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                )
            ],
        )
        captain_task = next(
            task for task in tasks
            if task.kind == ReviewTaskKind.TITLE_ROLE_ATTACHMENT
            and task.subject_key == "captain"
            and "spoken by kohaku" in task.prompt
        )

        assert "most likely refer to watanabe yō" in captain_task.prompt

    def test_address_like_prompt_filters_weak_corpus_title_owners(self):
        # Corpus fallback should not dump every one-off title holder into the
        # prompt. Weak tail owners make the semantic question noisier without
        # adding useful guidance.
        clusters = build_reference_clusters([
            ReferenceCandidate(
                document_anchor=DocumentAnchor(path="a.md"),
                reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
                surface="Captain",
                normalized="captain",
                anchor=SpanAnchor(path="a.md", span_ordinal=0, start_char=1, end_char=8),
                context_before="\"",
                context_after=", wait,\" Kohaku said.",
                in_quote=True,
                address_like=True,
                quote_speaker_key="kohaku",
                linked_entity_keys=["kohaku"],
            ),
        ])
        tasks = build_review_tasks(
            clusters,
            [],
            [
                CharacterSemanticSummary(
                    canonical_key="watanabe yō",
                    alias_keys=["watanabe", "yō"],
                    supporting_document_paths=["b.md"],
                    attached_title_counts={"captain": 6},
                    ambiguous_title_counts={},
                    attached_relation_counts={},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                ),
                CharacterSemanticSummary(
                    canonical_key="tsushima yoshiko",
                    alias_keys=["tsushima", "yoshiko"],
                    supporting_document_paths=["c.md"],
                    attached_title_counts={"captain": 5},
                    ambiguous_title_counts={},
                    attached_relation_counts={},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                ),
                CharacterSemanticSummary(
                    canonical_key="prosser",
                    alias_keys=[],
                    supporting_document_paths=["d.md"],
                    attached_title_counts={"captain": 1},
                    ambiguous_title_counts={},
                    attached_relation_counts={},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                ),
            ],
        )
        captain_task = next(
            task for task in tasks
            if task.kind == ReviewTaskKind.TITLE_ROLE_ATTACHMENT
            and task.subject_key == "captain"
            and "spoken by kohaku" in task.prompt
        )

        assert "watanabe yō" in captain_task.prompt
        assert "tsushima yoshiko" in captain_task.prompt
        assert "prosser" not in captain_task.prompt

    def test_address_like_prompt_prefers_dominant_corpus_title_owner(self):
        # When one non-speaker owner clearly dominates corpus title evidence,
        # the prompt should surface that likely owner directly instead of
        # keeping a weaker runner-up in the same question.
        clusters = build_reference_clusters([
            ReferenceCandidate(
                document_anchor=DocumentAnchor(path="a.md"),
                reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
                surface="Captain",
                normalized="captain",
                anchor=SpanAnchor(path="a.md", span_ordinal=0, start_char=1, end_char=8),
                context_before="\"",
                context_after=", wait,\" Kohaku said.",
                in_quote=True,
                address_like=True,
                quote_speaker_key="kohaku",
                linked_entity_keys=["kohaku"],
            ),
        ])
        tasks = build_review_tasks(
            clusters,
            [],
            [
                CharacterSemanticSummary(
                    canonical_key="watanabe yō",
                    alias_keys=["watanabe", "yō"],
                    supporting_document_paths=["b.md"],
                    attached_title_counts={"captain": 10},
                    ambiguous_title_counts={},
                    attached_relation_counts={},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                ),
                CharacterSemanticSummary(
                    canonical_key="tsushima yoshiko",
                    alias_keys=["tsushima", "yoshiko"],
                    supporting_document_paths=["c.md"],
                    attached_title_counts={"captain": 5},
                    ambiguous_title_counts={},
                    attached_relation_counts={},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                ),
            ],
        )
        captain_task = next(
            task for task in tasks
            if task.kind == ReviewTaskKind.TITLE_ROLE_ATTACHMENT
            and task.subject_key == "captain"
            and "spoken by kohaku" in task.prompt
        )

        assert "most likely refer to watanabe yō" in captain_task.prompt
        assert "tsushima yoshiko" not in captain_task.prompt

    def test_relation_prompt_uses_canonical_character_keys(self):
        # Local relation prompts should collapse alias-like candidates to their
        # canonical character keys before they reach review output.
        clusters = [
            ReferenceCluster(
                document_anchor=DocumentAnchor(path="doc.md"),
                reference_type=ReferenceCandidateType.BARE_RELATION_ROLE,
                normalized="mother",
                surface_forms=["mother"],
                occurrence_count=1,
                anchors=[],
                in_quote_count=0,
                address_like_count=0,
                speaker_entity_scores={},
                candidate_entity_scores={"tsushima": 1, "yoshiko": 1},
            )
        ]
        tasks = build_review_tasks(
            clusters,
            [],
            [
                CharacterSemanticSummary(
                    canonical_key="tsushima yoshiko",
                    alias_keys=["tsushima", "yoshiko"],
                    supporting_document_paths=["doc.md"],
                    attached_title_counts={},
                    ambiguous_title_counts={},
                    attached_relation_counts={},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                )
            ],
        )
        mother_task = next(
            task for task in tasks
            if task.kind == ReviewTaskKind.RELATION_ROLE_ATTACHMENT
            and task.subject_key == "mother"
        )

        assert "one of tsushima yoshiko" in mother_task.prompt
        assert "tsushima, yoshiko" not in mother_task.prompt

    def test_address_like_relation_prompt_prefers_dominant_corpus_owner(self):
        # Address-like relation roles can also rely on corpus ownership when
        # the local sentence only tells us who is speaking.
        clusters = build_reference_clusters([
            ReferenceCandidate(
                document_anchor=DocumentAnchor(path="a.md"),
                reference_type=ReferenceCandidateType.BARE_RELATION_ROLE,
                surface="Master",
                normalized="master",
                anchor=SpanAnchor(path="a.md", span_ordinal=0, start_char=1, end_char=7),
                context_before="\"",
                context_after=", please,\" Kohaku said.",
                in_quote=True,
                address_like=True,
                quote_speaker_key="kohaku",
                linked_entity_keys=["kohaku"],
            ),
        ])
        tasks = build_review_tasks(
            clusters,
            [],
            [
                CharacterSemanticSummary(
                    canonical_key="tsushima yoshiko",
                    alias_keys=["tsushima", "yoshiko"],
                    supporting_document_paths=["b.md"],
                    attached_title_counts={},
                    ambiguous_title_counts={},
                    attached_relation_counts={"master": 6},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                ),
                CharacterSemanticSummary(
                    canonical_key="watanabe yō",
                    alias_keys=["watanabe", "yō"],
                    supporting_document_paths=["c.md"],
                    attached_title_counts={},
                    ambiguous_title_counts={},
                    attached_relation_counts={"master": 2},
                    ambiguous_relation_counts={},
                    aggregate_attribution_count=0,
                    conflict_sources=[],
                ),
            ],
        )
        master_task = next(
            task for task in tasks
            if task.kind == ReviewTaskKind.RELATION_ROLE_ATTACHMENT
            and task.subject_key == "master"
        )

        assert "spoken by kohaku most likely refer to tsushima yoshiko" in master_task.prompt

    def test_address_like_cluster_demotes_speaker_from_candidate_ranking(self):
        # Direct address often targets someone other than the quote speaker.
        # When both appear as local candidates, the grouped review cluster
        # should not rank the known speaker first by default.
        clusters = build_reference_clusters([
            ReferenceCandidate(
                document_anchor=DocumentAnchor(path="doc.md"),
                reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
                surface="Captain",
                normalized="captain",
                anchor=SpanAnchor(path="doc.md", span_ordinal=0, start_char=1, end_char=8),
                context_before="\"",
                context_after=", wait,\" Kohaku said to Yo.",
                in_quote=True,
                address_like=True,
                quote_speaker_key="kohaku",
                linked_entity_keys=["kohaku", "yō"],
            ),
            ReferenceCandidate(
                document_anchor=DocumentAnchor(path="doc.md"),
                reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
                surface="captain",
                normalized="captain",
                anchor=SpanAnchor(path="doc.md", span_ordinal=0, start_char=30, end_char=37),
                context_before="Yo nodded at the ",
                context_after=".",
                in_quote=False,
                address_like=False,
                quote_speaker_key=None,
                linked_entity_keys=["kohaku"],
            ),
        ])
        captain_cluster = next(
            cluster for cluster in clusters
            if cluster.reference_type == ReferenceCandidateType.BARE_TITLE_ROLE
            and cluster.normalized == "captain"
        )

        assert list(captain_cluster.candidate_entity_scores) == ["yō", "kohaku"]


class TestConflictTyping:
    def test_alias_component_conflict_is_typed_as_component_pollution(self):
        # When a merged canonical inherits the conflicting category from an
        # absorbed component key rather than from its own exact surface, the
        # review layer should explain that difference explicitly.
        corpus = reconcile_document_entities([
            _make_record("a.md", "tsushima yoshiko", LexiconCategory.UNRESOLVED, resolved=False, confidence_score=0.20),
            _make_record("a.md", "tsushima", LexiconCategory.CHARACTER, confidence_score=0.70),
            _make_record("a.md", "yoshiko", LexiconCategory.CHARACTER, confidence_score=0.75),
            _make_record("b.md", "yoshiko", LexiconCategory.PLACE, confidence_score=0.40),
        ])

        conflicts = build_conflict_records(corpus.canonical_entities)

        assert len(conflicts) == 1
        assert conflicts[0].canonical_key == "tsushima yoshiko"
        assert conflicts[0].source == ConflictSource.COMPONENT_POLLUTION

    def test_exact_surface_conflict_is_typed_as_surface_level_disagreement(self):
        # When the exact same surface disagrees across documents, the conflict
        # should be reported as surface-level disagreement rather than alias
        # component pollution.
        corpus = reconcile_document_entities([
            _make_record("a.md", "meridian", LexiconCategory.GROUP, confidence_score=0.65),
            _make_record("b.md", "meridian", LexiconCategory.PLACE, confidence_score=0.62),
        ])

        conflicts = build_conflict_records(corpus.canonical_entities)

        assert len(conflicts) == 1
        assert conflicts[0].canonical_key == "meridian"
        assert conflicts[0].source == ConflictSource.SURFACE_LEVEL_DISAGREEMENT


class TestCharacterSummaries:
    def test_unique_title_cluster_attaches_to_one_character_summary(self):
        # A grouped title mention that points to one canonical character should
        # become positive attachment evidence in that character's semantic
        # summary instead of staying only as a flat review prompt.
        corpus = reconcile_document_entities([
            _make_record("a.md", "watanabe yō", LexiconCategory.CHARACTER, confidence_score=0.70),
            _make_record("a.md", "watanabe", LexiconCategory.CHARACTER, confidence_score=0.65),
            _make_record("a.md", "yō", LexiconCategory.CHARACTER, confidence_score=0.65),
        ])
        conflicts = build_conflict_records(corpus.canonical_entities)
        summaries = build_character_summaries(
            corpus.canonical_entities,
            [
                ReferenceCluster(
                    document_anchor=DocumentAnchor(path="a.md"),
                    reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
                    normalized="admiral",
                    surface_forms=["admiral"],
                    occurrence_count=3,
                    anchors=[],
                    in_quote_count=0,
                    address_like_count=0,
                    speaker_entity_scores={},
                    candidate_entity_scores={"watanabe yō": 3},
                )
            ],
            conflicts,
        )

        watanabe_summary = next(summary for summary in summaries if summary.canonical_key == "watanabe yō")
        assert watanabe_summary.attached_title_counts == {"admiral": 3}
        assert watanabe_summary.ambiguous_title_counts == {}

    def test_ambiguous_title_cluster_stays_ambiguous_in_character_summaries(self):
        # Shared titles are common in fiction. If a grouped title still points
        # to multiple canonicals, the summary should preserve that ambiguity
        # instead of attaching the title to one person prematurely.
        corpus = reconcile_document_entities([
            _make_record("a.md", "watanabe yō", LexiconCategory.CHARACTER, confidence_score=0.70),
            _make_record("a.md", "watanabe", LexiconCategory.CHARACTER, confidence_score=0.65),
            _make_record("a.md", "yō", LexiconCategory.CHARACTER, confidence_score=0.65),
            _make_record("a.md", "connall", LexiconCategory.CHARACTER, confidence_score=0.68),
        ])
        conflicts = build_conflict_records(corpus.canonical_entities)
        summaries = build_character_summaries(
            corpus.canonical_entities,
            [
                ReferenceCluster(
                    document_anchor=DocumentAnchor(path="a.md"),
                    reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
                    normalized="admiral",
                    surface_forms=["admiral"],
                    occurrence_count=2,
                    anchors=[],
                    in_quote_count=0,
                    address_like_count=0,
                    speaker_entity_scores={},
                    candidate_entity_scores={"watanabe yō": 1, "connall": 1},
                )
            ],
            conflicts,
        )

        watanabe_summary = next(summary for summary in summaries if summary.canonical_key == "watanabe yō")
        connall_summary = next(summary for summary in summaries if summary.canonical_key == "connall")
        assert watanabe_summary.attached_title_counts == {}
        assert connall_summary.attached_title_counts == {}
        assert watanabe_summary.ambiguous_title_counts == {"admiral": 2}
        assert connall_summary.ambiguous_title_counts == {"admiral": 2}

    def test_unique_relation_cluster_attaches_to_one_character_summary(self):
        # Relation clusters should feed the same character-centric summary
        # layer as titles so later semantic review can inspect stable kinship
        # evidence per canonical character.
        corpus = reconcile_document_entities([
            _make_record("a.md", "connall", LexiconCategory.CHARACTER, confidence_score=0.68),
        ])
        conflicts = build_conflict_records(corpus.canonical_entities)
        summaries = build_character_summaries(
            corpus.canonical_entities,
            [
                ReferenceCluster(
                    document_anchor=DocumentAnchor(path="a.md"),
                    reference_type=ReferenceCandidateType.BARE_RELATION_ROLE,
                    normalized="brother",
                    surface_forms=["brother"],
                    occurrence_count=2,
                    anchors=[],
                    in_quote_count=0,
                    address_like_count=0,
                    speaker_entity_scores={},
                    candidate_entity_scores={"connall": 2},
                )
            ],
            conflicts,
        )

        connall_summary = next(summary for summary in summaries if summary.canonical_key == "connall")
        assert connall_summary.attached_relation_counts == {"brother": 2}
        assert connall_summary.ambiguous_relation_counts == {}
