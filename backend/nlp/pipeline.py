"""
Pipeline orchestration for document and corpus NLP passes.

This module centralizes stage ordering so entrypoints do not duplicate
cross-stage wiring or accidentally drift in behavior.

.. code-block:: mermaid

    flowchart TD
        A[path + raw text] --> B[parse]
        B --> C[preprocess]
        C --> D[bootstrap using shared pre]
        D --> E[attribute_dialogue]
        E --> F[promote]
        F --> G[summarize_document_entities]
        G --> H[extract_reference_candidates]
        H --> I[DocumentPipelineResult]
        I --> J[Corpus aggregation loop]
        J --> K[CorpusPipelineResult]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.nlp.lexicon.bootstrap import BootstrapResult, bootstrap
from backend.nlp.parsing.document_parser import parse
from backend.nlp.parsing.preprocessing import PreprocessedDocument, preprocess
from backend.nlp.promotion.attribution import AttributionRecord, attribute_dialogue
from backend.nlp.promotion.promotion import PromotionResult, promote
from backend.nlp.reconciliation.document_entities import summarize_document_entities
from backend.nlp.semantic_review.review import extract_reference_candidates
from backend.nlp.harvesting.shared import TITLE_PREFIXES_LOWER
from backend.nlp.types import (
    DocumentEntityRecord,
    ParsedMarkdownDocument,
    PromotedEvidenceBundle,
    ReferenceCandidate,
)


@dataclass
class DocumentPipelineResult:
    """Complete deterministic outputs for one source document.

    Args:
        doc: Parsed span model for the source text.
        pre: Token, sentence, and quote preprocessing model.
        bootstrap_result: Lexicon convergence outputs.
        attribution_records: Quote speaker attributions.
        promotion_bundle: Promotion buckets and evidence windows.
        entity_records: Stable document entity summaries.
        reference_candidates: Deferred reference candidates.
    """

    doc: ParsedMarkdownDocument
    pre: PreprocessedDocument
    bootstrap_result: BootstrapResult
    attribution_records: list[AttributionRecord]
    promotion_bundle: PromotedEvidenceBundle
    promotion_result: PromotionResult
    entity_records: list[DocumentEntityRecord]
    reference_candidates: list[ReferenceCandidate]


@dataclass
class CorpusPipelineResult:
    """Aggregated deterministic outputs across multiple documents.

    Args:
        document_results: Per-document deterministic results in input order.
        entity_records: Flattened entity records from all documents.
        reference_candidates: Flattened reference candidates from all documents.
    """

    document_results: list[DocumentPipelineResult]
    entity_records: list[DocumentEntityRecord]
    reference_candidates: list[ReferenceCandidate]


def run_document_pipeline(path: str, raw_text: str) -> DocumentPipelineResult:
    """Run the deterministic pipeline stages for one document.

    Args:
        path: Source document path.
        raw_text: Raw source text.

    Returns:
        Complete deterministic outputs for this document.
    """
    doc = parse(path, raw_text)
    pre = preprocess(doc)
    bootstrap_result = bootstrap(doc, pre=pre)
    combined_titles_lower = TITLE_PREFIXES_LOWER | frozenset(
        title.lower() for title in bootstrap_result.induced_title_prefixes
    )
    attribution_records = attribute_dialogue(pre, bootstrap_result.clusters)
    promotion_result = promote(
        pre,
        bootstrap_result.clusters,
        bootstrap_result.lexicon,
        attribution_records,
        title_prefixes_lower=combined_titles_lower,
    )
    promotion_bundle = promotion_result.bundle
    entity_records = summarize_document_entities(
        pre,
        bootstrap_result.clusters,
        promotion_bundle,
        promotion_result.scores,
        promotion_result.classifications,
    )
    reference_candidates = extract_reference_candidates(pre, entity_records, attribution_records)
    return DocumentPipelineResult(
        doc=doc,
        pre=pre,
        bootstrap_result=bootstrap_result,
        attribution_records=attribution_records,
        promotion_bundle=promotion_bundle,
        promotion_result=promotion_result,
        entity_records=entity_records,
        reference_candidates=reference_candidates,
    )


def run_corpus_pipeline(paths: list[str]) -> CorpusPipelineResult:
    """Run deterministic document pipeline across many source documents.

    Args:
        paths: Ordered source document path strings.

    Returns:
        Aggregated corpus-level deterministic outputs.
    """
    document_results: list[DocumentPipelineResult] = []
    entity_records: list[DocumentEntityRecord] = []
    reference_candidates: list[ReferenceCandidate] = []
    for path in paths:
        result = run_document_pipeline(path, raw_text=Path(path).read_text(encoding="utf-8"))
        document_results.append(result)
        entity_records.extend(result.entity_records)
        reference_candidates.extend(result.reference_candidates)
    return CorpusPipelineResult(
        document_results=document_results,
        entity_records=entity_records,
        reference_candidates=reference_candidates,
    )
