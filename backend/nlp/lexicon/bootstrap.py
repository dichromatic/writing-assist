"""
Bootstrapped lexicon convergence loop.

Runs up to max_passes total passes over the document. Pass 0 uses only the
manuscript harvester. Each subsequent pass compiles the current lexicon into
an Aho-Corasick automaton, runs phrase matching over all spans to generate
additional candidates, merges them with the harvested candidates
(deduplicating by span position), and re-clusters. The loop terminates early
when no new normalised_phrase keys appear in the lexicon.

.. code-block:: mermaid

    flowchart TD
        A[ParsedMarkdownDocument] --> B[preprocess]
        B --> C[Pass 0: harvest_manuscript]
        C --> D[cluster_mentions]
        D --> E[induce_lexicon pass=0]
        E --> F{More passes allowed\nAND new entries?}
        F -->|Yes| G[compile_automaton]
        G --> H[match_text over all spans]
        H --> I[Deduplicate candidates]
        I --> J[cluster_mentions]
        J --> K[induce_lexicon pass=N]
        K --> F
        F -->|No| L[BootstrapResult]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.nlp.types import (
    BootstrappedLexiconEntry,
    Heading,
    MentionCandidate,
    MentionCluster,
    Paragraph,
    ParsedMarkdownDocument,
)
from backend.nlp.parsing.preprocessing import preprocess, PreprocessedDocument
from backend.nlp.harvesting.manuscript import harvest_manuscript
from backend.nlp.clustering.clustering import cluster_mentions
from backend.nlp.clustering.linking import link_clusters
from backend.nlp.lexicon.induction import induce_lexicon
from backend.nlp.lexicon.matcher import compile_automaton, match_text
from backend.nlp.lexicon.title_induction import TitleInductionDiagnostic, induce_title_prefixes
from backend.nlp.harvesting.shared import TITLE_PREFIXES


@dataclass
class BootstrapResult:
    """Result of the convergence loop.

    Args:
        lexicon: Final set of inducted lexicon entries after all passes.
        clusters: Final MentionCluster records from the last pass.
        candidates: Final deduplicated MentionCandidate records used to
            build the last clustering.
        passes_run: Total number of passes executed (always >= 1).
        new_entries_per_pass: Number of new normalised_phrase keys inducted
            in each pass. Index 0 = pass 0 total; index N = new keys in pass N.
            Length always equals passes_run.
    """

    lexicon: list[BootstrappedLexiconEntry]
    clusters: list[MentionCluster]
    candidates: list[MentionCandidate]
    passes_run: int
    new_entries_per_pass: list[int]
    induced_title_prefixes: frozenset[str] = field(default_factory=frozenset)
    title_induction_diagnostics: list[TitleInductionDiagnostic] = field(default_factory=list)


def _match_all_spans(
    automaton: 'ahocorasick.Automaton',
    pre: PreprocessedDocument,
    path: str,
) -> list[MentionCandidate]:
    """Run the automaton over every heading and paragraph span.

    SceneBreak spans are skipped - they have no text to match against.

    Args:
        automaton: A compiled Aho-Corasick automaton from compile_automaton.
        pre: The preprocessed document whose source spans are scanned.
        path: Document path for MentionCandidate anchor construction.

    Returns:
        All lexicon-matched MentionCandidate records, in document order.
    """
    doc = pre.source
    candidates: list[MentionCandidate] = []
    content_spans = sorted(
        [*doc.headings, *doc.paragraphs],
        key=lambda s: s.span_ordinal,
    )
    for span in content_spans:
        candidates.extend(match_text(automaton, span.text, path, span.span_ordinal, span.start_char))
    return candidates


def _deduplicate_candidates(candidates: list[MentionCandidate]) -> list[MentionCandidate]:
    """Remove candidates that cover the same span as an earlier candidate.

    When the harvest pass and the lexicon pass both find the same surface at
    the same position, only the first occurrence (harvest candidate) is kept.
    Deduplication is by (start_char, end_char) of the anchor.

    Args:
        candidates: Combined candidates from all passes, harvest-pass candidates
            first so they win over lexicon-pass candidates at the same position.

    Returns:
        Deduplicated candidates in input order.
    """
    seen: set[tuple[int, int]] = set()
    result: list[MentionCandidate] = []
    for c in candidates:
        key = (c.anchor.start_char, c.anchor.end_char)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


def bootstrap(
    doc: ParsedMarkdownDocument,
    max_passes: int = 3,
    pre: PreprocessedDocument | None = None,
) -> BootstrapResult:
    """Run the convergence loop and return the final lexicon and clusters.

    Pass 0 always runs. Each subsequent pass (up to max_passes - 1 additional
    passes) uses the lexicon from the previous pass to find additional
    candidates via Aho-Corasick phrase matching. The loop terminates early
    when no new normalised_phrase keys appear in the lexicon.

    Args:
        doc: A parsed Markdown document from the parsing stage.
        max_passes: Maximum total number of passes including pass 0. Must be
            >= 1. Set to 1 to disable the convergence loop (pass 0 only).
        pre: Optional preprocessed document. When provided, bootstrap reuses
            this object instead of preprocessing the parsed document again.

    Returns:
        BootstrapResult with the final lexicon, clusters, candidates, pass count,
        and per-pass new-entry counts.
    """
    pre = pre or preprocess(doc)

    # Pass 0: harvest without lexicon
    candidates = harvest_manuscript(pre)
    clusters = cluster_mentions(candidates)
    link_clusters(clusters, [], [], [])
    lexicon = induce_lexicon(clusters, doc.path, induction_pass=0)

    new_entries_per_pass: list[int] = [len(lexicon)]
    passes_run = 1

    prev_phrases: set[str] = {e.normalized_phrase for e in lexicon}

    for pass_idx in range(1, max_passes):
        if not lexicon:
            # No entries to match with - cannot improve.
            break

        automaton = compile_automaton(lexicon)
        lexicon_candidates = _match_all_spans(automaton, pre, doc.path)

        # Harvest candidates take priority when deduplicating.
        all_candidates = _deduplicate_candidates(candidates + lexicon_candidates)

        new_clusters = cluster_mentions(all_candidates)
        link_clusters(new_clusters, [], [], [])
        new_lexicon = induce_lexicon(new_clusters, doc.path, induction_pass=pass_idx)

        new_phrases: set[str] = {e.normalized_phrase for e in new_lexicon}
        new_count = len(new_phrases - prev_phrases)

        new_entries_per_pass.append(new_count)
        passes_run += 1
        candidates = all_candidates
        clusters = new_clusters
        lexicon = new_lexicon
        prev_phrases = new_phrases

        if new_count == 0:
            break  # Converged.

    induced_titles, title_diagnostics = induce_title_prefixes(
        clusters=clusters,
        candidates=candidates,
        lexicon=lexicon,
    )
    if induced_titles:
        expanded_prefixes = TITLE_PREFIXES | induced_titles
        reharvested_candidates = harvest_manuscript(pre, title_prefixes=expanded_prefixes)
        if lexicon:
            automaton = compile_automaton(lexicon)
            lexicon_candidates = _match_all_spans(automaton, pre, doc.path)
            all_candidates = _deduplicate_candidates(reharvested_candidates + lexicon_candidates)
        else:
            all_candidates = reharvested_candidates
        clusters = cluster_mentions(all_candidates)
        link_clusters(clusters, [], [], [])
        induced_lexicon = induce_lexicon(clusters, doc.path, induction_pass=passes_run)
        induced_phrases: set[str] = {entry.normalized_phrase for entry in induced_lexicon}
        new_entries_per_pass.append(len(induced_phrases - prev_phrases))
        lexicon = induced_lexicon
        prev_phrases = induced_phrases
        candidates = all_candidates
        passes_run += 1

    return BootstrapResult(
        lexicon=lexicon,
        clusters=clusters,
        candidates=candidates,
        passes_run=passes_run,
        new_entries_per_pass=new_entries_per_pass,
        induced_title_prefixes=induced_titles,
        title_induction_diagnostics=title_diagnostics,
    )
