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

- `documentation/phase-0.md`

## Phase 1

Goal:

- import a project root
- let the user assign directory roles
- discover project files from those mappings
- parse Markdown into the first useful span model
- render imported documents in the editor

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
- empty directories are either omitted or clearly marked
- heuristic suggestions are stable and deterministic

Done when:

- the frontend can request candidate directories for a project root
- the backend returns structured import candidates

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
- discovered documents inherit the correct role-derived type
- stable ordering is preserved

Done when:

- the current hardcoded discovery spike has been replaced

### Phase 1.6: Markdown parse to span model

Deliverables:

- first parser output for:
  - headings
  - paragraphs
  - sections
- stable document-relative span ordering

TDD applies:

- yes

Behavior to test:

- headings split sections correctly
- paragraphs are extracted correctly across blank lines
- empty lines do not become spans
- mixed heading/paragraph content remains ordered

Done when:

- a Markdown document can be converted into a basic span list

### Phase 1.7: Editor loads imported document

Deliverables:

- file tree based on discovered documents
- load selected document into CodeMirror
- selection-aware state for later Analysis/Editing/Ideation actions

TDD applies:

- partial

Test:

- backend/document loading logic should be tested
- basic UI wiring does not need exhaustive TDD

Done when:

- a configured project can be opened and a manuscript file can be viewed in the editor

### Phase 1 completion criteria

- project root import works
- directory roles are user-defined and persisted
- file discovery uses those mappings
- Markdown parsing produces the first span model
- imported documents can be opened in the editor
- Phase 1 documentation exists in `documentation/phase-1-*.md`

## Phase 2

Goal:

- introduce mode-aware chat and pass orchestration for `Analysis`, `Editing`, and `Ideation`

### Phase 2.1: Core pass contracts

Deliverables:

- `PassRequest`
- `PassResult`
- `ContextBundle`
- mode-specific allowed output types

TDD applies:

- yes

Behavior to test:

- `Analysis` cannot emit draft changes
- `Editing` emits bounded draft changes
- `Ideation` emits idea outputs, not direct edits by default

### Phase 2.2: Chat thread model

Deliverables:

- `ChatThread`
- thread attachment to selection/document/project scope
- stored mode and context policy per thread

TDD applies:

- yes

### Phase 2.3: Orchestrator path

Deliverables:

- route frontend requests into mode-aware pass execution
- basic provider stub path using the existing healthcheck-style bridge

TDD applies:

- yes

### Phase 2.4: Frontend mode-aware chat UI

Deliverables:

- chat panel
- mode switcher
- current scope display
- display of pass outputs

TDD applies:

- partial

### Phase 2 completion criteria

- user can start a thread in each mode
- backend receives structured pass requests
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

### Phase 4.3: Validator pass

Deliverables:

- validator request/response contract
- rewrite validation before draft promotion where enabled

TDD applies:

- yes

### Phase 4.4: Consistency checks

Deliverables:

- canon consistency pass
- terminology consistency pass

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

1. Rewrite the current indexing tests around user-defined directory mappings.
2. Replace hardcoded discovery/classification code with config-driven discovery.
3. Define the first persisted schema for projects and directory mappings.
4. Build the import UI for assigning directory roles.
