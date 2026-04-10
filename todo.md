# TODO

This file breaks the implementation plan into concrete execution slices.

It is intended to make feature work small enough that:

- we know what the next deliverable is
- we know what behavior needs tests
- we know what can be treated as wiring or scaffolding

## Execution Rules

- Use red/green/refactor TDD for domain logic, parsing, indexing, retrieval, persistence rules, and feature behavior.
- Do not force TDD for trivial framework wiring, passive UI layout, or mechanical config changes.
- If anything unexpected happens, stop and notify the user before continuing.
- After each completed phase, write a technical implementation note under `documentation/`.

## Phase 0

Status:

- completed

Artifacts:

- containerized development environment
- Rust workspace scaffold
- SvelteKit frontend scaffold
- Tauri desktop scaffold
- frontend-to-Tauri healthcheck

Documentation:

- `documentation/phase-0-workspace-setup.md`

## Phase 1

Goal:

- import a project root
- let the user assign directory roles
- discover project files from those mappings
- parse Markdown into the first useful span model
- render imported documents in the editor
- define project context source categories before mode-aware chat consumes them

### Phase 1.1: Import configuration model

Deliverables:

- `ProjectDirectoryRole`
- `ProjectDirectoryMapping`
- `ProjectImportCandidate`
- `ProjectConfig` updated to include directory-role mappings

TDD applies:

- yes

Behavior to test:

- arbitrary directory names can be mapped to roles
- exactly one `primary_manuscript` role is required
- `reference`, `notes`, and `ignore` roles behave distinctly
- heuristics only suggest defaults and do not override explicit mappings

Done when:

- indexing code no longer depends on hardcoded folder names
- role configuration can drive downstream discovery behavior

### Phase 1.2: Candidate directory scan

Deliverables:

- scan a chosen project root for candidate directories
- generate suggestion records for the import UI
- provide rationale for suggestions like `contains .md files` or `named chapters`

TDD applies:

- yes

Behavior to test:

- immediate child directories are discovered correctly
- root-level Markdown files create a `.` root candidate
- empty directories are either omitted or clearly marked
- hidden/app directories such as `.git` and `.writing-assist` are not import candidates
- heuristic suggestions are stable and deterministic

Done when:

- the frontend can request candidate directories for a project root
- the backend returns structured import candidates, including a root candidate when root-level Markdown exists

### Phase 1.3: Import UI

Deliverables:

- project root picker
- import screen listing candidate directories
- role assignment controls
- validation that blocks continuation until one `primary_manuscript` is chosen

TDD applies:

- partial

Test:

- frontend behavior tests only where the UI enforces project rules
- no need to TDD pure layout or cosmetic work

Done when:

- a user can assign roles and submit a valid import configuration

### Phase 1.4: Persist project import configuration

Deliverables:

- initial SQLite schema for projects and directory mappings
- persistence path for saving import configuration
- load path for reopening an existing configured project

TDD applies:

- yes

Behavior to test:

- mappings persist correctly
- reopening a project reuses stored mappings
- invalid mappings are rejected

Done when:

- import configuration survives app restart and can drive indexing

### Phase 1.5: File discovery from configured mappings

Deliverables:

- discover Markdown files from mapped directories
- classify documents based on configured directory role, not folder name
- ignore unmapped or explicitly ignored directories

TDD applies:

- yes

Behavior to test:

- only enabled mapped directories are scanned
- only Markdown files are indexed
- root-level Markdown files work through a safe `.` mapping
- child directory mappings override broad `.` mappings by specificity
- hidden/app directories are skipped when recursively discovering from broad mappings
- supported Markdown extensions include `.md`, `.markdown`, `.mdown`, and uppercase variants
- discovered documents inherit the correct role-derived type
- stable ordering is preserved

Done when:

- the current hardcoded discovery spike has been replaced

### Phase 1.6: Markdown parse to span model

Deliverables:

- parser output for:
  - headings
  - paragraphs
  - sections
  - explicit scene-break spans
  - first-class scene objects
  - byte and character offsets
  - whitespace-normalized sidecar text
  - section-boundary metadata
- configurable paragraph parsing:
  - strict blank-line mode
  - conservative heuristic mode
- stable document-relative span ordering

TDD applies:

- yes

Behavior to test:

- headings split sections correctly
- paragraphs are extracted correctly across blank lines
- empty lines do not become spans
- mixed heading/paragraph content remains ordered
- explicit scene breaks split sections and create scenes
- non-ASCII text has correct byte and character offsets
- normalized sidecar text does not mutate source text
- strict paragraph mode disables the conservative heuristic

Done when:

- a Markdown document can be converted into spans, sections, scenes, offsets, and normalized sidecars

### Phase 1.7: Editor loads imported document

Deliverables:

- file tree based on discovered documents
- load selected document into CodeMirror

TDD applies:

- partial

Test:

- backend/document loading logic should be tested
- basic UI wiring does not need exhaustive TDD

Done when:

- a configured project can be opened and a manuscript file can be viewed in the editor

### Phase 1.8: Document workspace and selection targeting

Deliverables:

- extract loaded-document rendering into a dedicated document workspace component
- keep `ProjectImportPanel` focused on import/open/load actions
- expose CodeMirror selection state from the Markdown editor
- map selected character ranges to parsed span ordinals
- surface selected text and overlapping spans for later Analysis/Editing/Ideation actions

TDD applies:

- partial

Test:

- selection-to-span mapping should be tested because it is domain/feature behavior
- basic Svelte component layout and event wiring do not need exhaustive TDD

Done when:

- loaded documents render through a document workspace component
- text selection updates app state
- selection state includes selected text, character range, and overlapping span ordinals
- selection state is lifted to the workspace parent so Phase 2 chat/task UI can consume it
- Phase 2 can target the current selection/window without depending on import UI internals

### Phase 1.9: Project context source taxonomy

Deliverables:

- define the distinction between broad directory roles and document-level context source types
- add core types for first-class guide sources:
  - prose guideline
  - style guide
  - critique rubric
  - rewrite guide
  - custom guide
- add core types for first-class reference sources:
  - story summary
  - world summary
  - character bible
  - timeline
  - terminology
  - research
  - custom reference
- define how `guide`, `reference`, and `note` sources enter `Analysis`, `Editing`, and `Ideation`
- decide which parts must be implemented before Phase 2 task contracts and which parts can wait for Phase 3 retrieval/memory

TDD applies:

- yes once source classification or task-context policy becomes executable business logic

Test:

- core taxonomy serialization should be tested
- default mode-specific context inclusion rules should be tested
- future tests should cover source classification once it is implemented

Done when:

- `implementation.md` and `todo.md` define context source semantics clearly enough to shape Phase 2 `TaskRequest` and `ContextBundle`
- `core` exposes context source taxonomy types and default mode policy helpers
- context-source default inclusion checks activation and review state as well as source kind
- prose/style/critique guides are not treated as ordinary notes
- story/world/character/timeline/terminology bibles are not treated as untyped blobs
- notes remain opt-in or retrieval-based, not automatically injected into prompts

### Phase 1.10: Phase 1 hardening sweep

Deliverables:

- validate directory mappings as safe project-relative paths
- support root-level Markdown projects with a normalized `.` mapping
- preserve saved mappings that are temporarily absent from a candidate rescan
- skip hidden/app directories during broad discovery
- clarify that parser-emitted spans are heading, paragraph, and scene-break spans, while sections/scenes/windows are target categories
- lift document selection target state for Phase 2 chat/task handoff
- expose a context-source default inclusion helper that respects activation and review state

TDD applies:

- yes for mapping safety, discovery, context policy, and mapping-retention behavior
- partial for component event handoff

Done when:

- Phase 1 review findings have concrete tests or documented scope boundaries
- Phase 2 can build task contracts without relying on unsafe mappings or private editor state

### Phase 1 completion criteria

- project root import works
- directory roles are user-defined and persisted
- directory mappings are normalized and cannot escape the project root
- root-level Markdown manuscripts can be imported through the `.` mapping
- file discovery uses those mappings
- Markdown parsing produces the first span model
- imported documents can be opened in the editor
- current editor selection can be mapped to parsed spans
- current editor selection is available outside the document component for chat/task handoff
- project context source categories are defined before Phase 2 task contracts
- Phase 1 documentation exists in numbered subphase files such as `documentation/phase-1.1-*.md`

## Phase 2

Goal:

- introduce mode-aware chat and task orchestration for `Analysis`, `Editing`, and `Ideation`

### Phase 2.1: Core task contracts

Status:

- completed

Deliverables:

- `SelectionTarget`
- `TaskRequest`
- `TaskResult`
- `ContextBundle`
- mode-specific allowed output types and context-source policies
- stable task IDs and schema/version fields for future persistence
- explicit references to selected guide/reference/note source IDs or paths

TDD applies:

- yes

Behavior to test:

- `Analysis` cannot emit draft changes
- `Editing` emits bounded draft changes
- `Ideation` emits idea outputs, not direct edits by default
- task requests can target the current selection/span/section/scene/window set from Phase 1.8/1.10
- context-source defaults use the taxonomy from Phase 1.9 and the activation/review gate from Phase 1.10
- notes are not included by default

Done when:

- core task contracts compile without depending on provider/Rig types
- mode output constraints are test-covered
- task context inputs can represent selected spans and selected/pinned context sources

Documentation:

- `documentation/phase-2.1-mode-aware-llm-task-contracts.md`

### Phase 2.2: Context bundle assembly stub

Deliverables:

- build a local `ContextBundle` from:
  - selected document path
  - selected text
  - selected span ordinals
  - selected/pinned context sources
- no embeddings or LLM calls yet
- no automatic retrieval yet

TDD applies:

- yes

Behavior to test:

- selected text and span ordinals are preserved
- allowed context sources are included according to mode policy
- disallowed/default-excluded notes stay out unless explicitly selected

Done when:

- orchestrator can construct a deterministic context bundle from local app state

### Phase 2.3: Chat thread model

Deliverables:

- `ChatThread`
- thread attachment to selection/document/project scope
- stored mode and context policy per thread

TDD applies:

- yes

### Phase 2.4: Orchestrator path

Deliverables:

- route frontend requests into mode-aware task execution
- basic provider stub path using the existing healthcheck-style bridge
- return deterministic placeholder outputs before real model calls

TDD applies:

- yes

### Phase 2.5: Frontend mode-aware chat UI

Deliverables:

- chat panel
- mode switcher
- current scope display
- display of task outputs
- show current selected text/span target from the document workspace
- show selected/pinned context sources once available

TDD applies:

- partial

### Phase 2 completion criteria

- user can start a thread in each mode
- backend receives structured task requests
- outputs are mode-correct

## Phase 3

Goal:

- add reviewable project memory and hybrid retrieval

### Phase 3.1: Entity extraction

Deliverables:

- extract candidate entities from imported docs
- persist candidates

TDD applies:

- yes

### Phase 3.2: Fact and summary candidates

Deliverables:

- candidate facts
- candidate summaries
- review state and stale state

TDD applies:

- yes

### Phase 3.3: Memory review UI

Deliverables:

- approve/reject summaries
- approve/reject facts

TDD applies:

- partial

### Phase 3.4: Hybrid retrieval

Deliverables:

- metadata retrieval
- lexical retrieval
- fact/entity retrieval
- vector retrieval abstraction

TDD applies:

- yes

### Phase 3 completion criteria

- approved memory can be reused in context bundles
- stale memory is excluded until refreshed

## Phase 4

Goal:

- make editing reviewable and canon-aware

### Phase 4.1: Draft change model

Deliverables:

- `DraftChange`
- draft lifecycle states
- diff payload storage

TDD applies:

- yes

### Phase 4.2: Diff review flow

Deliverables:

- accept/reject draft changes
- apply accepted edits to Markdown files

TDD applies:

- yes for backend file-application logic
- partial for UI

### Phase 4.3: Validator task

Deliverables:

- validator request/response contract
- rewrite validation before draft promotion where enabled

TDD applies:

- yes

### Phase 4.4: Consistency checks

Deliverables:

- canon consistency task
- terminology consistency task

TDD applies:

- yes

### Phase 4 completion criteria

- bounded edits are reviewable
- accepted edits write back correctly
- consistency checks only use approved memory

## Phase 5

Goal:

- harden provider access and settings

### Phase 5.1: API-key providers

Deliverables:

- provider config storage
- API-key setup flow

TDD applies:

- yes for config/validation logic
- no for basic settings UI wiring

### Phase 5.2: Experimental subscription bridge

Deliverables:

- isolated adapter boundary
- explicit unstable/personal-use labeling

TDD applies:

- yes for adapter-selection and failure isolation logic

### Phase 5.3: Provider settings UX

Deliverables:

- switch provider
- inspect provider state
- surface failures clearly

TDD applies:

- partial

### Phase 5 completion criteria

- provider configuration is stable
- adapter failures do not corrupt project state

## Immediate Next Tasks

1. Start Phase 2.2 context bundle assembly stub with TDD.
2. Convert the frontend `DocumentSelectionTarget` shape into the core `SelectionTarget` contract.
3. Build deterministic local context bundles from selected/pinned context sources.
4. Keep embeddings, retrieval, and model/provider calls out of Phase 2.2.
