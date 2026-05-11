# TODO

This file breaks the implementation plan into concrete execution slices.

It is intended to make feature work small enough that:

- we know what the next deliverable is
- we know what behavior needs tests
- we know what can be treated as wiring or scaffolding

## Execution Rules

- Test the decision, not the implementation. Write tests that encode non-obvious rules whose breakage would be invisible. Skip tests for self-evident implementation details.
- Use pytest + selective Hypothesis: Hypothesis for preprocessing invariants (offset stability, normalization idempotence), pytest for harvesting rules and promotion boundaries.
- Derive regression tests from inspection log failures, not from preemptive coverage.
- Do not force TDD for trivial framework wiring, passive UI layout, or mechanical config changes.
- If anything unexpected happens, stop and notify the user before continuing.
- After each completed phase, write a technical implementation note under `documentation/`.

---

## Python NLP Pipeline Port

Goal:

- port the Rust evidence pipeline to Python
- deliver a runnable, inspectable pipeline before connecting FastAPI routes, database persistence, or provider integration
- implement improvements not present in the Rust reference: TF-IDF scoring, dialogue attribution, graded confidence scores

Work in this order. Each subphase depends on the previous one.

### Phase P.1: Pipeline types and data model

Status:

- completed

Deliverables:

- dataclasses in `backend/nlp/types.py` for all pipeline input/output types:
  - `ParsedMarkdownDocument`, `Heading`, `Paragraph`, `SceneBreak`, `Section`, `Scene`
  - `PreprocessedDocument`, `Token`, `Sentence`, `QuoteSpan`, `StructuralMarker`
  - `MentionCandidate`, `StructuredFieldCandidate`, `DefinitionCandidate`, `SectionSummarySeed`
  - `MentionCluster`
  - `BootstrappedLexiconEntry`
  - `PromotedEvidenceBundle`, `PromotedCandidate`, `ReviewOnlyCandidate`, `SuppressedCandidate`
  - anchor types: `DocumentAnchor`, `SpanAnchor`, `SectionAnchor`
- `stable_hash_id` utility on all record types
- version tag on all types for future serialization compatibility

Out of scope:

- implementation logic in any pipeline module
- FastAPI routes, database schema, frontend types

TDD applies:

- yes for hash stability and any invariants expressed as constructors

Behavior to test:

- `stable_hash_id` is deterministic for typical inputs
- anchor fields are required; types cannot be constructed without source provenance
- dataclass field names match what downstream stages consume

Done when:

- all pipeline stages can express their input and output types by importing from `types.py`
- no circular imports between `types.py` and any stage module

Documentation:

- `documentation/python-p1-pipeline-types.md`

### Phase P.2: Markdown parser

Status:

- completed

Deliverables:

- `backend/nlp/parsing/markdown_parser.py`:
  - parse Markdown text into `ParsedMarkdownDocument`
  - emit `Heading`, `Paragraph`, and `SceneBreak` spans with byte and character offsets
  - derive sections from heading boundaries
  - derive scenes from explicit scene-break markers (`---`)
  - produce whitespace-normalized sidecar text without mutating source text
  - include section-boundary metadata (file-start, heading, scene-break)

Out of scope:

- preprocessing, tokenization, sentence segmentation
- entity extraction
- front matter or YAML header parsing

TDD applies:

- yes

Behavior to test:

- headings correctly split sections
- paragraphs are extracted across blank lines without creating empty spans
- non-ASCII text produces correct byte and character offsets
- normalized sidecar text does not mutate or shift source offsets
- scene-break markers produce both a `SceneBreak` span and a scene boundary
- mixed heading/paragraph content is ordered correctly
- a document with no headings or scene breaks produces a single section

Done when:

- a Markdown file produces a `ParsedMarkdownDocument` with correct spans, offsets, sections, and scenes

Documentation:

- `documentation/python-p2-markdown-parser.md`

### Phase P.3: Preprocessing

Status:

- completed

Deliverables:

- `backend/nlp/parsing/preprocessing.py`:
  - Unicode normalization: smart quotes to straight quotes, em-dashes to hyphens, apostrophes, ellipses
  - stable tokenizer with preserved source offsets post-normalization
  - sentence boundary detection that handles dialogue and headings correctly without collapsing them
  - explicit `QuoteSpan` detection from normalized text
  - structural markers: heading, list item, and scene-break tagging

Out of scope:

- harvesting or entity detection
- syntactic parsing or dependency roles
- pretrained models of any kind

TDD applies:

- yes
- Hypothesis for offset stability under normalization and normalization idempotence

Behavior to test:

- token source offsets round-trip correctly through Unicode normalization
- sentence segmentation avoids merging dialogue attribution lines with preceding sentences
- quote spans are recoverable from manuscripts with varied quoting styles
- heading and list-item structural markers are tagged correctly for planning and reference files
- normalization is idempotent: running it twice produces the same result
- non-ASCII characters do not shift downstream token offsets

Done when:

- `ParsedMarkdownDocument` produces a `PreprocessedDocument` with stable, inspectable token and sentence boundaries

Documentation:

- `documentation/python-p3-preprocessing.md`

### Phase P.4: Harvesting shared infrastructure and manuscript archetype

Status:

- completed

Deliverables:

- `backend/nlp/harvesting/shared.py`:
  - `stable_hash_id`
  - anchor merge utilities
  - occurrence and feature merge utilities
  - title prefix list: Mr, Mrs, Ms, Miss, Dr, Prof, Captain, Cpt, Lady, Lord, Sir, Dame, Admiral, General, Sergeant, ...
  - label lists for field detection
  - NLTK-backed stopword set
  - structural suppression helpers: sentence-initial singleton check, bullet-start singleton check, field-label position check, heading-only singleton check

- `backend/nlp/harvesting/manuscript.py`:
  - extract `MentionCandidate` from prose spans using title/possessive/capitalization patterns
  - suppress sentence-initial unsupported singletons, dialogue-internal noise, and stopword tokens
  - attach document path and span/section anchors to every candidate

Out of scope:

- other archetype harvesters
- clustering or promotion
- cross-document merge

TDD applies:

- yes

Behavior to test:

- title prefix patterns (`Captain Aldous`) produce candidates; bare title tokens alone do not
- possessive forms (`Aldous's`) attach to the correct base mention
- sentence-initial capitalized singletons with no further support are suppressed
- stopwords are filtered without any project-specific word lists
- every emitted candidate includes document path and span anchor
- suppression does not remove recurring names that have support in other spans
- `shared.py` exports are the only source of title prefix lists and stopwords; no duplicates in harvester modules

Done when:

- `harvesting/shared.py` and `harvesting/manuscript.py` are implemented
- manuscript archetype produces `MentionCandidate` records with correct anchors and suppression behavior on a test document

Documentation:

- `documentation/python-p4-harvesting-shared-and-manuscript.md`

### Phase P.5: Evidence clustering

Status:

- completed

Deliverables:

- `backend/nlp/clustering/clustering.py`:
  - group `MentionCandidate` records into `MentionCluster` by normalized surface form
  - strip possessives, normalize whitespace and case for the grouping key
  - merge anchors, occurrence counts, and support features across clustered candidates

- `backend/nlp/clustering/linking.py`:
  - cross-link clusters to related `StructuredFieldCandidate` and `DefinitionCandidate` records
  - cross-link clusters to `SectionSummarySeed` where the cluster surface appears in the seed context

Out of scope:

- lexicon induction
- alias resolution across different surface forms
- promotion

TDD applies:

- yes

Behavior to test:

- `Aldous`, `Aldous's`, and `Captain Aldous` cluster under the same normalized key
- clusters preserve all source anchors from merged candidates
- occurrence count reflects total mentions across all merged candidates
- linking attaches the correct field or definition candidate without losing its anchor
- normalization grouping is deterministic and stable across processing order
- a candidate that appears in multiple sections preserves all section anchors

Done when:

- harvested `MentionCandidate` records from `manuscript.py` fold into `MentionCluster` records with correct grouping and cross-links

Documentation:

- `documentation/python-p5-evidence-clustering.md`

### Phase P.6: Bootstrapped lexicon and convergence loop

Status:

- completed

Deliverables:

- `backend/nlp/lexicon/induction.py`:
  - induce `BootstrappedLexiconEntry` records from clustered evidence
  - record provenance: source anchors, occurrence counts, archetypes seen in, rule sources
  - manuscript induction is mention-led
  - reference/taxonomy induction is definition-led
  - planning/dossier induction is field-led

- `backend/nlp/lexicon/matcher.py`:
  - compile induced entries into a `pyahocorasick` automaton
  - run phrase matching over normalized preprocessed text
  - emit matched `MentionCandidate` records with span offsets

- `backend/nlp/lexicon/bootstrap.py`:
  - bounded document-level convergence loop (max passes configurable, default 3)
  - re-run harvesters with the compiled lexicon until no new entries appear or max-pass cap is reached
  - track pass metrics: new entries per pass, convergence delta

Out of scope:

- cross-document lexicon merge
- treating lexicon entries as approved canon memory
- LLM-driven alias merging

TDD applies:

- yes

Behavior to test:

- a document with no user-provided vocabulary bootstraps lexicon entries from repeated surfaces
- later passes produce additional matches that were not found in pass 0
- common stopwords are not inducted as lexicon entries
- structurally weak singletons (sentence-initial, bullet-start) do not survive induction
- titled names survive induction; bare title tokens alone do not
- convergence loop terminates at max-pass cap even if new entries still appear
- matcher produces candidates with correct source offsets

Done when:

- the convergence loop produces a stable per-document bootstrapped lexicon
- second-pass harvest improvements are visible in inspection output

Documentation:

- `documentation/python-p6-bootstrapped-lexicon.md`

### Phase P.7: Evidence promotion, TF-IDF scoring, and dialogue attribution

Status:

- completed

Deliverables:

- `backend/nlp/promotion/attribution.py`:
  - detect post-quote attribution: `"..." , NameSpan said/asked/replied/...`
  - detect pre-quote attribution: `NameSpan said/asked/... , "..."`
  - attach speaker attribution signal to the nearest `MentionCandidate` or cluster
  - filter dialogue-internal surface forms not attributed to any speaker

- `backend/nlp/promotion/scoring.py`:
  - compute per-document and per-section term frequencies
  - TF-IDF score per cluster as a continuous signal
  - combine signals into a graded deterministic confidence score:
    - rule tier: seed lexicon entry > titled pattern > capitalization only
    - title/honorific presence
    - possessive occurrence frequency
    - speech-verb co-occurrence near quote spans (from attribution pass)
    - scene dispersion: appears across multiple scenes is stronger
    - TF-IDF specificity

- `backend/nlp/promotion/promotion.py`:
  - classify clusters into `PromotedCandidate`, `ReviewOnlyCandidate`, or `SuppressedCandidate` using graded scores
  - construct `PromotedEvidenceBundle` with all three buckets and source provenance
  - define `EvidenceWindow` type: entity-centric context slice for retrieval (first introduction, attributed dialogue, high-activation nearby events)

Out of scope:

- retrieval implementation
- vector search or embeddings
- provider calls
- final canon/alias resolution

TDD applies:

- yes

Behavior to test:

- post-quote and pre-quote attribution patterns produce correct speaker signals on test sentences
- scene-dispersed clusters score higher than single-scene singletons under the same rule tier
- TF-IDF downweights names that appear at background frequency across all sections
- graded confidence scores are deterministic given the same input
- ambiguous middle-confidence clusters land in `review_only` rather than being forced into `promoted` or `suppressed`
- suppressed candidates include a traceable suppression reason
- `EvidenceWindow` carries the anchor of its source span

Done when:

- the pipeline produces `PromotedEvidenceBundle` records with graded confidence
- speaker attribution signals are visible in inspection output for a manuscript document with dialogue

Documentation:

- `documentation/python-p7-promotion-scoring-attribution.md`

### Phase P.8: Pipeline wiring and inspection tool

Status:

- not started

Deliverables:

- `backend/nlp/pipeline.py`:
  - orchestrate all stages in order: parser -> preprocessing -> harvesting -> clustering -> lexicon -> promotion
  - accept a document path and archetype label, return a `PromotedEvidenceBundle`
  - expose per-stage intermediate output for inspection

- `backend/inspect.py`:
  - CLI tool: `uv run --project backend python backend/inspect.py path/to/doc.md`
  - print per-stage output with configurable verbosity (spans, tokens, candidates, clusters, lexicon entries, promoted/review/suppressed bundles)
  - structured JSON output option for later FastAPI integration

Out of scope:

- FastAPI routes
- database persistence
- frontend integration

TDD applies:

- partial: pipeline wiring is integration, not the decision layer; test that stage output types feed into downstream stage inputs cleanly

Behavior to test:

- running the pipeline on a manuscript document produces a non-empty `PromotedEvidenceBundle`
- all promoted candidates have at least one source anchor
- stage outputs feed into downstream stage inputs without type errors or missing fields
- the max-pass convergence cap prevents infinite loops on degenerate input

Done when:

- `inspect.py` on a Markdown document produces readable per-stage output
- pipeline wiring has no silent data loss between stages

Documentation:

- `documentation/python-p8-pipeline-wiring-and-inspection.md`

### Phase P.9: Remaining archetype harvesters

Status:

- not started

Deliverables:

- `backend/nlp/harvesting/dossier.py`:
  - field-led harvesting: alias fields, participant fields, role fields
  - section structure: profile sections as evidence units
  - suppress prose-approach and tone vocabulary

- `backend/nlp/harvesting/planning.py`:
  - block-structure-led: scene/beat/outline headings, goal/outcome fields
  - participant and named entity extraction from field-grounded positions
  - suppress editorial and planning-descriptor vocabulary

- `backend/nlp/harvesting/reference.py`:
  - definition-led: glossary-like headings, acronym/expansion patterns, explicit term blocks
  - stricter about descriptive singletons and heading fragments than manuscript harvesting

- `backend/nlp/harvesting/loose_note.py`:
  - conservative wrapper using reference-style suppression
  - avoid promoting bullet-start singletons or editorial asides as candidates

- `backend/nlp/harvesting/dispatch.py`:
  - lookup table mapping archetype label to harvester function
  - no if/elif chain; raise on unknown archetypes

Out of scope:

- cross-archetype semantic merging
- project-level identity resolution across documents

TDD applies:

- yes for archetype-specific suppression rules

Behavior to test:

- dossier alias fields produce candidate records; prose-adjacent tone words do not
- planning participant fields survive; planning-descriptor words (`compelling`, `brutal`, `tense`) do not
- reference descriptive heading fragments are rejected; explicit term blocks survive
- loose notes do not promote bullet-start singletons
- dispatch routes each archetype to exactly one harvester
- all harvesters produce candidates using utilities from `shared.py` only; no inline title prefix lists or stopword sets

Done when:

- all five archetype harvesters and `dispatch.py` are implemented
- full pipeline runs on a document of each archetype type
- `inspect.py` shows archetype-appropriate evidence for each document type

### Python NLP pipeline completion criteria

- all pipeline stage modules are implemented and importable without circular imports
- `inspect.py` produces readable per-stage output for all five supported archetypes
- `types.py` covers all input/output types with source anchors required on every record
- `harvesting/shared.py` is the single source for shared constants, utilities, title prefix lists, and stopword sets
- no archetype module duplicates title prefix lists, stopword sets, or merge logic
- convergence loop terminates deterministically within max-pass cap
- test suite covers: offset stability, normalization idempotence, suppression rules, archetype policy, promotion boundary, attribution patterns
- inspection output on a real manuscript document shows meaningful promoted candidates with traceable suppression reasons

---

## Near-Term Follow-Up Work

These items sit on top of the current manuscript semantic-review work and the
new dossier experiment scaffold. They are intentionally smaller and more
concrete than the older archetype-harvester phases below.

### Manuscript handoff artifact

Status:

- completed

Goal:

- persist a machine-readable manuscript handoff bundle that matches the current
  question-first semantic-review boundary
- render the manuscript inspection report from that persisted artifact instead
  of relying only on ad hoc report assembly

Deliverables:

- a persisted `ManuscriptReviewBundle` JSON artifact shape built from the
  current manuscript pipeline outputs
- report rendering that reads from the manuscript handoff artifact
- explicit inclusion of:
  - document-level entity records
  - reference clusters
  - conflict records
  - review tasks
  - grouped suppressed evidence
  - absorbed surface families
  - ranked reference and attachment alternatives

Out of scope:

- changing the manuscript semantic-review boundary from review questions to
  proposal-style outputs
- forcing manuscript documents into the same record-unit structure used by the
  dossier experiment
- LLM semantic resolution

TDD applies:

- yes

Behavior to test:

- the persisted artifact preserves grouped suppressed evidence under stable
  local containers
- the persisted artifact preserves ranked reference alternatives and speaker
  context without hiding them in prompt text only
- absorbed and aliased surface families remain explicit in the artifact
- the rendered manuscript report can be produced from the persisted artifact
  without losing current review-question content

Done when:

- manuscript inspection output can be regenerated from a stored
  `ManuscriptReviewBundle`
- the manuscript handoff is machine-readable in the same broad architectural
  style as the dossier review scaffold while keeping manuscript-specific
  analysis units

Documentation:

- `documentation/python-p54-manuscript-review-bundle.md`

### Document status metadata

Status:

- deferred

Goal:

- add optional per-document status metadata that influences source authority
  and review weighting without changing extraction family selection

Deliverables:

- support for optional document status metadata on any document type
- default behavior of `primary_canon` when no explicit status is set
- support for both in-document metadata and sidecar-manifest metadata
- conflict handling when both metadata sources disagree
- integration of document status into provenance and authority layers

Out of scope:

- creating new extraction families solely because of status labels
- using status to choose the extraction pipeline
- blocking ingestion when metadata sources disagree

TDD applies:

- yes

Behavior to test:

- documents without explicit status default to `primary_canon`
- matching in-document and sidecar status values are accepted
- conflicting in-document and sidecar values downgrade to `draft_unknown`
  and emit a reviewable metadata conflict
- folder names can emit soft hints but do not silently override explicit or
  default status
- status influences authority weighting without changing document-family
  routing

Done when:

- document status is available as provenance metadata across manuscripts,
  structured notes, and later document types
- status can affect downstream authority and review behavior without changing
  segmentation or extraction family selection

Documentation:

- `documentation/document-status-metadata-notes.md`

### Retrieval object and manuscript editing flow notes

Status:

- documented

Goal:

- preserve the current retrieval architecture discussion while extraction work
  continues
- keep the target object model clear without finalizing the database schema too
  early
- model how explicit manuscript-editing retrieval should behave before
  frontend or storage implementation begins

Deliverables:

- a retrieval-object architecture note covering claim units, evidence,
  grouping, ambiguity, result levels, retrieval channels, modes, reasons,
  ranking, scope widening, and diagnostics
- a manuscript-editing retrieval-flow note covering explicit invocation,
  intent gating, skipped retrieval, channel execution, chat context packaging,
  manual pinning, and source-use tracking

Out of scope:

- implementing retrieval
- implementing frontend panes
- finalizing database tables
- choosing vector, relational, or hybrid storage
- finalizing the canonical world model

TDD applies:

- no

Behavior to test:

- none yet; this is a planning artifact

Done when:

- future extraction, LLM-pass, retrieval, and database work can reference the
  same retrieval-object target without relying on chat history

Documentation:

- `documentation/retrieval-object-architecture-notes.md`
- `documentation/manuscript-editing-retrieval-flow-notes.md`

---

## Phase 4: FastAPI server and editing

Goal:

- wire the Python NLP pipeline into FastAPI routes
- add SQLite persistence for memory records
- make editing reviewable in the frontend

### Phase 4.1: FastAPI server and NLP routes

Deliverables:

- FastAPI app in `backend/main.py`
- `POST /analyze`: accepts document path and archetype, runs pipeline, returns `PromotedEvidenceBundle`
- `GET /health`: connectivity check for frontend
- structured error responses

TDD applies:

- yes for route input validation and error handling
- partial for pipeline integration

### Phase 4.2: SQLite persistence for memory records

Deliverables:

- `aiosqlite`-backed persistence for entity candidates, reviewable facts, reviewable summaries, review states, stale states
- store APIs: save pending candidates, list by state, approve/reject, mark stale on source change
- migration schema versioning

TDD applies:

- yes

Behavior to test:

- pending records persist and reload correctly
- approve/reject transitions persist
- stale records are excluded from reusable memory queries
- source references survive round-trip

### Phase 4.3: Draft change model and review flow

Deliverables:

- `DraftChange` type with lifecycle states and diff payload
- CodeMirror decoration overlays for draft changes
- accept/reject controls
- accepted edits write back to Markdown files
- stale related editor anchors after accepted edits

TDD applies:

- yes for backend file-application logic
- partial for UI

### Phase 4.4: Consistency checks

Deliverables:

- canon consistency task using approved memory only
- terminology consistency task using approved memory only

TDD applies:

- yes

### Phase 4 completion criteria

- FastAPI server runs in Docker and responds to frontend fetch calls
- pipeline results persist to SQLite
- bounded edits are reviewable and accepted edits write back correctly
- consistency checks only use approved memory

---

## Phase 5: Provider integration and semantic consolidation

Goal:

- harden provider access and settings
- add LLM-backed semantic consolidation over promoted evidence bundles

### Phase 5.1: API-key provider setup

Deliverables:

- provider config storage
- API-key setup flow
- provider abstraction that keeps domain logic independent of provider SDK types

TDD applies:

- yes for config validation
- no for basic settings UI

### Phase 5.2: Experimental subscription bridge

Deliverables:

- isolated adapter boundary
- explicit unstable/personal-use labeling in UI and code

TDD applies:

- yes for adapter-selection and failure isolation logic

### Phase 5.3: Provider-backed semantic consolidation

Deliverables:

- semantic filter pipeline that consumes promoted evidence bundles, not raw mention clusters
- schema-validated promotion into typed candidates: `EntityProfileCandidate`, `RelationshipCandidate`, `TimelineEventCandidate`, `StoryArcCandidate`, `WorldRuleCandidate`, `TerminologyCandidate`
- explicit rejection path for borderline or noisy evidence
- alias-resolution policy that remains reviewable and source-linked
- archetype-specific provider tasks: manuscript entity/alias consolidation, reference terminology/world-rule consolidation, planning-note participant/goal/relationship consolidation
- provider-output validation: reject unknown evidence IDs, malformed rejection payloads, and weak abbreviations promoted without explicit expansion

TDD applies:

- yes for schema validation, promotion rules, and failure handling

Behavior to test:

- provider output cannot bypass schema validation
- promoted candidates preserve source anchors from harvested evidence
- semantically rejected evidence does not enter reusable memory
- alias merges remain inspectable and reversible

Done when:

- harvested evidence can be turned into typed, reviewable candidates without trusting raw provider output directly

### Phase 5.4: Memory review UI

Deliverables:

- review UI for semantically consolidated candidates
- approve/reject controls through backend API calls
- visible stale state
- no automatic use of pending memory in task context

TDD applies:

- partial

Done when:

- user can review promoted memory before it is allowed into retrieval or context selection

### Phase 5 completion criteria

- provider configuration is stable
- semantic consolidation is schema-validated and source-linked
- adapter failures do not corrupt project state
- memory review UI gates all machine-derived candidates before retrieval
