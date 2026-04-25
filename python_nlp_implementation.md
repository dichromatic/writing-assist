# Python NLP Implementation Plan

## Context

This document captures architectural decisions and implementation plans for
porting the writing-assist NLP pipeline from Rust to Python.

The full application stack (Tauri desktop shell + SvelteKit frontend) stays as
is. Only the Rust backend crates are being replaced with a Python backend.

## Architecture Decisions

### Backend integration

- Python backend runs as a local FastAPI server
- SvelteKit frontend calls it via `fetch` instead of `invoke`
- Tauri stays thin: window management and native dialogs (file picker, etc.)
- Rust in `src-tauri` shrinks to just the native shell

### Focus order

Work on the NLP pipeline first, specifically the deterministic end, before
wiring up FastAPI routes, Tauri integration, or the database layer.

## Python Stack

| Concern | Library |
|---|---|
| Multi-pattern matching | `pyahocorasick` |
| Unicode-aware regex | `regex` |
| Stopwords | NLTK stopwords corpus |
| Markdown parsing | custom (port of current parser) |
| Everything else | custom |

No spaCy. The pipeline is intentionally deterministic and archetype-aware in
ways that fight against spaCy's statistical defaults. The design philosophy
(explicit rules, stable offsets, no pretrained models) translates more cleanly
to lightweight custom Python.

## Pipeline Stages

The pipeline has seven stages with explicit input/output types at each
boundary.

```
ParsedMarkdownDocument
    ↓  [markdown parser]
PreprocessedDocument
    ↓  [preprocessing]
MentionCandidate, StructuredFieldCandidate, DefinitionCandidate, SectionSummarySeed
    ↓  [evidence harvesting — archetype-aware]
MentionCluster
    ↓  [evidence clustering]
BootstrappedLexiconEntry
    ↓  [lexicon induction]
MentionCandidate (second pass)
    ↓  [exact-phrase matching via aho-corasick]
    convergence loop until stable or max passes
PromotedEvidenceBundle
    ↓  [evidence promotion]
```

New stage to add (not in the Rust implementation):

```
MentionCluster + QuoteSpan
    ↓  [TF-IDF scoring]
    graded confidence scores per cluster
    ↓  feeds into promotion as continuous signal
```

## Module Structure

Avoid the Rust god-file problem (`evidence_harvesting.rs` is 1592 lines).
Separate archetype families into their own modules. Share infrastructure
explicitly.

Proposed layout:

```
nlp/
  pipeline.py               # top-level orchestration: run all stages in order
  types.py                  # shared dataclasses: all input/output types

  parsing/
    markdown_parser.py      # Markdown → ParsedMarkdownDocument
    preprocessing.py        # ParsedMarkdownDocument → PreprocessedDocument

  harvesting/
    shared.py               # shared utilities: stable_hash_id, merge logic,
                            # title prefix list, label lists, stopwords
    manuscript.py           # Manuscript archetype harvester
    dossier.py              # DossierProfile archetype harvester
    planning.py             # StoryPlanning archetype harvester
    reference.py            # TaxonomyReference + ExpositoryWorldArticle
    loose_note.py           # LooseNote archetype harvester (conservative wrapper)
    dispatch.py             # archetype → harvester function lookup table

  clustering/
    clustering.py           # evidence clustering
    linking.py              # cross-link clusters to fields, definitions, seeds

  lexicon/
    induction.py            # bootstrapped lexicon entry induction
    matcher.py              # compile + run aho-corasick exact-phrase matcher
    bootstrap.py            # bounded document-level convergence loop

  promotion/
    scoring.py              # TF-IDF scoring + graded confidence calculation
    promotion.py            # PromotedEvidenceBundle construction
    attribution.py          # dialogue attribution heuristics (quote → speaker)

  tests/
    nlp/
      test_preprocessing.py     # offset stability, normalization (+ Hypothesis)
      test_harvesting.py        # suppression rules, archetype policy
      test_clustering.py        # normalization grouping, possessive stripping
      test_lexicon.py           # survival rules, convergence behavior
      test_promotion.py         # promotion boundary, graded confidence
      test_attribution.py       # quote attribution heuristics

  inspect.py                    # CLI inspection tool
  pyproject.toml
```

Tests encode decisions: if a suppression rule changes, the test that locked
that decision fails. Regression tests are written from inspection log failures,
not speculatively.

Key rules:
- Each archetype harvester imports from `shared.py` — no duplicating title
  prefix lists or merge utilities across files
- `dispatch.py` is a lookup table, not an `if/elif` chain scattered through
  the pipeline
- `pipeline.py` is the only file that orchestrates across stages
- No class hierarchies — plain functions with typed inputs/outputs

## Improvements Over the Rust Implementation

### 1. TF-IDF scoring (new stage, high value)

Add a scoring pass after clustering, before promotion.

Purpose:
- Downweight names that appear so frequently they carry no retrieval signal
- Surface terms that are rare globally but frequent in one scene or chapter
- Improve `SectionSummarySeed` quality with TF-IDF-weighted term selection

Implementation:
- Compute per-document and per-section term frequencies during preprocessing
- Feed TF-IDF scores into `scoring.py` as a continuous signal
- Promotion uses graded scores instead of boolean flags

### 2. Dialogue attribution (new stage, addresses manuscript noise)

Add basic speaker attribution from detected quote spans.

The pipeline already detects quote spans. Attribution adds:
- Post-quote pattern: `"..." , NameSpan said/asked/...`
- Pre-quote pattern: `NameSpan said/asked/... , "..."`

Purpose:
- Strengthen entity candidacy for names near attribution verbs
- Filter dialogue-internal noise that is not attributed to anyone
- Separate what characters say from narrator text

Implemented in `attribution.py`. Result feeds into harvesting as an additional
support signal, not as a standalone output type.

### 3. Graded confidence scores (replaces binary promotion)

Upgrade promotion from three buckets (`promoted / review_only / suppressed`)
to a graded deterministic confidence score per cluster.

Scoring signals (all deterministic):
- Rule tier: seed lexicon > titled pattern > capitalization only
- Title/honorific presence
- Possessive occurrence frequency
- Co-occurrence with speech verbs near quotes (from attribution pass)
- Scene dispersion: appears across multiple scenes = stronger
- TF-IDF specificity

Purpose:
- LLM gating: call the LLM on ambiguous middle-confidence candidates only
- Review queue prioritization
- Retrieval ranking without vector search

### 4. Scene-aware evidence windows (contract for retrieval)

Plan evidence window generation as the retrieval interface, even if retrieval
is not implemented yet.

A window is a compact, entity-centric context slice:
- First introduction of an entity in a scene
- Dialogue attributed to an entity
- High-activation nearby events (fight, reveal, travel, arrival)

The retrieval layer should consume windows, not raw spans. Define the window
type now so Phase 3.8 retrieval has a stable input contract.

## What to Leave Alone

- Seedless bootstrap: better than author-provided lists. Keep it.
- Deferring apposition/epithet detection and syntactic parsing: confirmed
  correct by research.
- Archetype split philosophy: correct for fiction.
- aho-corasick for multi-pattern matching: `pyahocorasick` is the direct
  Python equivalent.

## Cross-Cutting Rules

- Every evidence record preserves document path + span/section anchors
- Shared constants (title prefixes, label lists, stopwords) live in
  `harvesting/shared.py` only — never duplicated across archetype modules
- `stable_hash_id` is one function in `shared.py`
- Merge utilities (anchors, occurrences, features) are one set of functions
  in `shared.py`
- No pretrained models, no statistical NER
- No project-specific deny-lists derived from example logs

## Priority Order

1. `types.py` — define all dataclasses first, nothing else compiles cleanly
   without them
2. `parsing/markdown_parser.py` — port of the current Rust parser
3. `parsing/preprocessing.py` — Unicode normalization, tokenization, sentence
   segmentation, quote spans, structural markers
4. `harvesting/shared.py` — shared utilities, title prefix list, stopwords
5. `harvesting/manuscript.py` — highest value archetype to validate pipeline
6. `clustering/clustering.py` + `clustering/linking.py`
7. `lexicon/induction.py` + `lexicon/matcher.py` + `lexicon/bootstrap.py`
8. `promotion/attribution.py` + `promotion/scoring.py` + `promotion/promotion.py`
9. `pipeline.py` — wire all stages together
10. Remaining archetype harvesters (dossier, planning, reference, loose_note)

## Open Questions

- What is the right max-pass cap for the document-level bootstrap loop?
  (currently 3 in Rust)
- Should the Python port include an inspection log equivalent (the Rust
  parser-log-tool), or rely on structured logging through FastAPI?
- Project-level cross-document lexicon merge: define the `ProjectLexicon`
  interface and output types now, but do not implement merge policy until
  per-document inspection logs from a real multi-document project justify the
  policy choices. Keeping the interface flexible also allows adding new
  document archetype categories without breaking the merge contract.

## Resolved Decisions

| Decision | Choice | Reason |
|---|---|---|
| Convergence loop execution model | Synchronous | CPU-bound, async buys nothing; wrap in `run_in_executor` when FastAPI needs it non-blocking |
| Inspection log | CLI tool first (`inspect.py`), structured logging added later | Development feedback loop is the priority; the two don't conflict |
| Cross-document lexicon merge | Define `ProjectLexicon` interface now, implement merge policy later | Need per-document log data to judge policy; interface must stay flexible for new archetype categories |
| Python backend location | `backend/` at project root | Clear convention, pairs naturally with frontend, standard FastAPI layout |
| Testing framework | pytest + selective Hypothesis | Hypothesis for preprocessing invariants (offset stability, normalization idempotence); pytest for harvesting rules and promotion boundaries |
| Testing discipline | Test the decision, not the implementation | Write tests that encode non-obvious rules whose breakage would be invisible; skip self-evident implementation details; derive regression tests from inspection log failures, not from preemptive coverage |
