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

Current note:

- P.1 through P.7 describe the manuscript evidence pipeline and are complete.
- P.8 and P.9 remain listed as historical pipeline-port phases, but the active
  non-manuscript work now runs through `structured_review`.
- Use `documentation/structured-review-reference-and-flow.md` as the terminology
  reference for document type, record family, document status, review bundles,
  LLM task packets, database proposals, and index-database handoff.

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

- completed, historical

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

Current state:

- `backend/inspect.py` exists and is the active manuscript inspection tool.
- `backend/nlp/pipeline.py` is now the canonical deterministic orchestration
  module.
- Entry points should call `run_document_pipeline` or `run_corpus_pipeline`
  instead of reassembling stage wiring ad hoc.
- Structured non-manuscript inspection now uses
  `inspect_structured_records.py` and `backend/nlp/experiments/structured_review/`.

### Phase P.9: Remaining archetype harvesters

Status:

- superseded for current non-manuscript work

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

Current state:

- `backend/nlp/harvesting/dossier.py`, `planning.py`, `reference.py`,
  `loose_note.py`, and `dispatch.py` are placeholders.
- The active path is no longer separate harvester modules per document type.
- The active path is:
  - classify corpus file document type
  - parse file into spans
  - segment by shape into structured-review record families
  - preserve deterministic seed bundles
- `ClaimUnit` exists as a structured-review local projection, but the current
  convergence plan treats it as transitional. Future database insertion should
  use shared database proposal objects.
- Do not revive this phase without first proving that shape-based
  structured-review record families are insufficient.

### Python NLP pipeline completion criteria

Current note:

- These criteria belong to the older archetype-harvester plan.
- For active structured-review work, use the near-term structured-review
  metadata, validation, and LLM handoff sections below.

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
structured-review scaffold. They are intentionally smaller and more
concrete than the older archetype-harvester phases below.

### Structured review terminology and flow reference

Status:

- documented

Goal:

- preserve the current terminology and data-flow decisions so future sessions
  do not re-invent or confuse document type, record family, document status,
  source authority, review bundles, LLM task packets, and database proposals

Deliverables:

- glossary for corpus file, document type, record family, document status,
  source authority, deterministic seed bundle, record review bundle,
  manuscript review bundle, LLM task packet, database proposal, proposal state,
  review state, approval state, and insertability state
- Mermaid data-flow diagram from corpus file to index database insertion
- current coverage matrix for manuscripts, story planning, world context,
  vignettes, locations, and character backgrounds
- explicit next implementation order

Out of scope:

- changing code
- finalizing database tables

TDD applies:

- no

Behavior to test:

- none, documentation only

Done when:

- the terminology reference can be used as the source of truth for future
  structured-review implementation work

Documentation:

- `documentation/structured-review-reference-and-flow.md`
- `documentation/review-bundle-llm-task-convergence-design.md`

### Structured review document type metadata

Status:

- completed

Goal:

- add first-class corpus-file document type metadata to structured-review
  artifacts and future database proposals before broader LLM evaluation

Deliverables:

- document type enum or typed value for `manuscript`, `vignette`,
  `story_planning`, `world_context`, `location`, and `character_background`
- path or manifest based classifier for the current example corpus
- `document_type` on structured-review diagnostics, review bundles, legacy
  prompt packets, and local claim projections
- reports that show document type separately from record family

Out of scope:

- using document type as the record segmentation family
- `.docx` ingestion
- database schema finalization

TDD applies:

- yes

Behavior to test:

- `examples/story planning/...` classifies as `story_planning`
- `examples/world context/...` classifies as `world_context`
- `examples/vignettes/...` classifies as `vignette`
- `examples/locations/...` classifies as `location` once ingested
- `examples/character backgrounds/...` classifies as `character_background`
- record family remains shape-based after document type is attached
- local claim projections preserve both `document_type` and `source_family`

Done when:

- structured-review JSON and logs expose document type for every processed file
- local claim projections can distinguish a `reference_section` from world
  context, location, character background, and story planning sources

Documentation:

- `documentation/structured-review-reference-and-flow.md`
- `documentation/python-p69-structured-review-document-type-metadata.md`

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
  structured-review experiment
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
  style as the structured-review scaffold while keeping manuscript-specific
  analysis units

Documentation:

- `documentation/python-p54-manuscript-review-bundle.md`

### Document status metadata

Status:

- completed

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
- status influences authority weighting without changing record-family routing

Done when:

- document status is available as provenance metadata across structured notes
  and is ready to be reused for manuscript bundle metadata parity
- status can affect downstream authority and review behavior without changing
  segmentation or extraction family selection

Documentation:

- `documentation/python-p70-document-status-metadata.md`
- `documentation/structured-review-reference-and-flow.md`

### Review bundle and LLM task convergence cleanup

Status:

- next

Goal:

- reconcile manuscript and structured-record review bundles before broader LLM
  or database work
- make review bundles deterministic-only
- move provider execution and task packets into a shared LLM task layer

Deliverables:

- remove LLM prompt and response fields from `RecordReviewBundle`
- remove `--run-llm` from `inspect_structured_records.py`
- move reusable provider transport logic toward `backend/nlp/llm_tasks/`
- keep inspection CLIs deterministic and artifact-oriented
- update structured-review reports so they no longer render legacy LLM slots
- mark `ClaimUnit` as transitional or retire it when database proposals exist

Out of scope:

- creating database tables
- running provider calls
- normalizing LLM results
- approving or canonicalizing proposals
- refactoring `backend/nlp/types.py` into a package

TDD applies:

- yes

Behavior to test:

- structured-review bundle construction still preserves deterministic seed
  bundles, subject guesses, and fact candidates
- deterministic structured-review reports no longer include LLM slot state
- structured-review CLI no longer runs provider calls
- manuscript and structured-record inspection remain deterministic
- old record-specific LLM tests are removed or rewritten around shared task
  packet generation

Done when:

- both manuscript and structured-record extraction stop at deterministic review
  artifacts
- live LLM work is no longer coupled to `RecordReviewBundle`
- the next phase can introduce shared `LLMTaskPacket` artifacts cleanly

Documentation:

- `documentation/structured-review-reference-and-flow.md`
- `documentation/review-bundle-llm-task-convergence-design.md`

### Shared LLM task packet layer

Status:

- blocked by review-bundle convergence cleanup

Goal:

- generate shared structured LLM task packets from both manuscript and
  structured-record review bundles without running a provider

Deliverables:

- shared `LLMTaskPacket` type
- shared task selection diagnostics
- structured-record task builder for `record_fact_extraction`
- manuscript task builders for entity profile, reference attachment, and
  category resolution tasks
- task-packet JSON artifacts from both inspection paths
- task-selection report artifacts from both inspection paths

Out of scope:

- provider execution
- LLM result normalization
- database insertion

TDD applies:

- yes

Behavior to test:

- task packets carry bounded evidence snippets plus source anchors
- skipped source objects emit selection diagnostics
- suppressed evidence can be included when locally relevant and labeled
- task packets reference schema ids rather than embedding provider-specific
  prompt text

Done when:

- manuscript and structured-review CLIs can emit shared-schema LLM task packet
  artifacts without provider calls

Documentation:

- `documentation/review-bundle-llm-task-convergence-design.md`

### Database proposal projection and readiness validation

Status:

- blocked by shared LLM task packet layer

Goal:

- project deterministic manuscript and structured-record outputs into a shared
  database proposal family and validate insertability

Deliverables:

- shared `DatabaseProposal` type
- indexing diagnostics for unsupported or not-yet-normalized observations
- deterministic structured-record claim projection into database proposals
- deterministic manuscript projection for entity profile, alias link,
  reference attachment, category resolution, and open review question proposals
- insertability validator for database proposal envelope fields, evidence, and
  payload presence
- proposal JSON artifacts from both inspection paths

Out of scope:

- final database tables
- canonical entity approval
- LLM result normalization
- relationship, timeline, or world-rule insertion

TDD applies:

- yes

Behavior to test:

- proposals preserve source document paths, source object ids, document type,
  document status, authority, evidence anchors, and evidence quotes
- review state, approval state, and insertability state are separate
- unsupported hint kinds become diagnostics rather than insertable proposals
- deterministic and future LLM-derived proposals can coexist without merging

Done when:

- both manuscript and structured-record artifacts can emit database-shaped
  proposal records that pass or fail insertability validation explicitly

Documentation:

- `documentation/review-bundle-llm-task-convergence-design.md`

### Shared LLM task runner and handoff probe

Status:

- blocked by shared LLM task packets and database proposal validation

Goal:

- run a small, deliberately chosen LLM probe through the shared task runner to
  test the handoff contract before broad corpus evaluation

Deliverables:

- fixed probe set of representative manuscript and structured-record task
  packets
- shared provider runner that accepts one or more task-packet files
- LLM task result artifacts
- deterministic normalization from accepted task result payloads into database
  proposals or diagnostics
- short inspection summary describing whether the handoff contract worked

Out of scope:

- corpus-wide LLM benchmark
- final extraction quality judgment
- canonical merge or entity resolution

TDD applies:

- partial, only for harness behavior and validation plumbing

Behavior to test:

- probe runner does not send unsupported task families to the LLM
- failed records do not fail the whole probe
- completed LLM results become reviewable database proposals or diagnostics
- validation diagnostics are produced for completed, failed, and skipped tasks

Done when:

- we can inspect whether the shared LLM task path produces database-shaped
  proposal records
  without pretending the quality question is settled

Documentation:

- `documentation/review-bundle-llm-task-convergence-design.md`

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

- a retrieval-object architecture note covering database proposals,
  transitional claim units, evidence,
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
