# Writing Assist v1 Implementation Plan

## Summary

Build a personal, local-first desktop writing workspace for Markdown projects using `Tauri + SvelteKit + CodeMirror + Rust + SQLite + Rig`. The app combines a primary chat/discussion surface with structured editorial tools, organized into three first-class modes: `Analysis`, `Editing`, and `Ideation`.

The manuscript folder remains canonical, but the app uses an internal draft layer for AI suggestions and human review. Project import is configuration-driven: the user chooses which directories contain primary manuscript content and which directories provide supporting reference or notes material. All model work is span-bounded, retrieval is hybrid and source-linked, and all machine-derived memory is review-gated before reuse.

## Naming Principles

- Prefer readable, product/domain-oriented names over abstract engineering shorthand.
- Use terminology that makes sense for writing assistance workflows, not generic LLM infrastructure terms.
- Rename unclear concepts early, before they spread across code, documentation, and UI copy.
- Keep names consistent across `implementation.md`, `todo.md`, phase documentation, Rust modules, and frontend types.
- Examples of preferred terminology:
  - use `task` instead of `pass`
  - use `task context selection` instead of `context assembly`
  - keep `ContextBundle` as the output type for the selected context attached to a task

## Implementation Changes

### Product and interaction model

- Make the main UI a hybrid workspace:
  - document tree + editor
  - primary chat/discussion panel
  - comments/diffs/context/memory side panels
- Add three first-class conversation modes:
  - `Analysis`: critique, explanation, consistency checks, no direct manuscript mutation
  - `Editing`: bounded rewrite suggestions that emit draft changes only
  - `Ideation`: constrained brainstorming over current selection/window and approved memory, no direct edits by default
- Default all chat sessions to `current selection/window` scope, with explicit expansion to nearby text, scene, chapter, and approved project memory.
- Treat active writing guides, prose guidelines, and critique rubrics as first-class context sources rather than generic reference notes.
- Preserve structured non-chat actions as shortcuts into the same backend task system:
  - comment on selection
  - rewrite selection
  - run task on scene
  - check consistency

### Frontend architecture

- Use `SvelteKit` as the frontend framework inside a Tauri desktop shell.
- Use `CodeMirror` as the v1 editor because Markdown files are the primary source format and paragraph/window diffs map naturally to text editing.
- Keep the editor integration behind an internal editor abstraction so a later `ProseMirror`/`Tiptap` frontend can replace the editing surface without changing backend contracts.
- Treat the current Phase 2 `ModeAwareChatPanel` as a prototype task surface, not the final locked frontend architecture.
- Move toward a `WriterWorkspace` shell rather than a side-chat wrapper:
  - project tree and document navigation
  - primary manuscript editor pane
  - optional reference editor panes for bibles, guides, outlines, and notes
  - intelligence hub for mode-aware chat, task outputs, context inspection, and thread anchors
  - knowledge rail for guide/reference/note context controls
  - draft review layer rendered over editor content
- Track focus and selection through explicit workspace state:
  - `focusedPaneId`
  - a selection registry keyed by pane ID
  - a derived `activeTaskTarget` that becomes the default input for `TaskRequest` construction
- Keep multi-pane layout pragmatic at first:
  - support a small set of fixed or lightly configurable layouts before building a full window manager
  - support pinning project files into reference panes once the pane registry exists
- Treat early span ordinal anchors as session-local:
  - use document path, character offsets, and span ordinals in Phase 2
  - add document content hashes and revision metadata before relying on anchors across edits
  - add durable span IDs or revalidated anchor records before accepting edits changes target text
- Render `DraftChange` output as non-destructive review UI:
  - Phase 2 can display suggestions
  - Phase 4 owns CodeMirror decoration overlays, accept/reject commands, filesystem mutation, and stale anchor handling
- Make the knowledge rail the explicit control surface for context sources:
  - guide/reference/note toggles should feed `explicitly_selected_source_paths`
  - backend task context selection must still apply mode policy, activation policy, and review-state gates
- Implement frontend modules for:
  - project/file navigation
  - Markdown editor with character selection mapped to parsed spans
  - document workspace for loaded Markdown documents
  - chat panel with mode-aware controls
  - comments rail and jump-to-span behavior
  - diff review and accept/reject flow
  - context inspector showing exact retrieved context
  - memory review UI for facts and summaries

### Frontend workspace architecture spec

The target frontend is a Writer's IDE, not a document editor with a side-chat wrapper. It should coordinate editor state, context controls, task execution, draft review, and chat references through explicit workspace state.

#### Workspace shell

The top-level UI should move toward a `WriterWorkspace` shell with these primary regions:

- project tree and document navigation
- primary manuscript editor pane
- optional reference editor panes for bibles, guides, outlines, and notes
- intelligence hub for chat, task outputs, context inspection, and thread anchors
- knowledge rail for guide/reference/note context controls
- draft review layer rendered over editor content

Likely component/module split:

```txt
src/lib/workspace/
  WriterWorkspace.svelte
  workspaceState.ts
  paneRegistry.ts
  selectionRegistry.ts
  draftReviewState.ts
  contextSourceState.ts
  anchorState.ts

src/lib/components/
  PrimaryEditorPane.svelte
  ReferenceEditorPane.svelte
  IntelligenceHub.svelte
  KnowledgeRail.svelte
  DraftChangeOverlay.svelte
```

The exact file names can change during implementation, but the responsibility boundaries should remain clear.

#### Focus and selection state

The frontend needs global focus and selection state because model tasks can originate from any editor pane.

Recommended state shape:

```ts
type PaneId = string;

type PaneSelection = {
  paneId: PaneId;
  documentPath: string;
  selectedText: string;
  startChar: number;
  endChar: number;
  overlappingSpanOrdinals: number[];
  primarySpanOrdinal: number | null;
  documentContentHash?: string;
};

type ActiveTaskTarget = {
  sourcePaneId: PaneId;
  documentPath: string;
  selectedText: string;
  startChar: number;
  endChar: number;
  anchors: Array<{ kind: 'span' | 'section' | 'scene' | 'window'; ordinal: number }>;
  documentContentHash?: string;
};

type WorkspaceState = {
  focusedPaneId: PaneId | null;
  activeSelectionPaneId: PaneId | null;
  panes: WorkspacePane[];
  selectionByPaneId: Record<PaneId, PaneSelection>;
};
```

`focusedPaneId` tracks general UI focus. `activeSelectionPaneId` tracks the editor pane that last produced a taskable selection. This distinction is required because the user will commonly select manuscript text and then move focus into the intelligence hub to run a task.

`activeTaskTarget` should be derived from the latest active editor selection, not from generic UI focus. The UI should return no task target if no editor selection has ever been activated rather than guessing from an arbitrary pane.

#### Span identity and stability

Early frontend anchors can use document path, character offsets, and parsed span ordinals, but those anchors are only session-local. Span ordinals can shift after edits.

Anchor stability should progress in phases:

- Phase 2: use document path, character offsets, and span ordinals
- Phase 3: include content hashes and parsed document revision metadata in targets
- Phase 4: introduce durable span IDs or revalidated anchor records before accepted edits mutate target text

Multi-span selections must preserve all overlapping span ordinals instead of collapsing to only the primary span.

#### Multi-pane layout

The workspace should support these pane types:

```ts
type WorkspacePane =
  | { paneType: 'primary_editor'; paneId: PaneId; documentPath: string | null }
  | { paneType: 'reference_editor'; paneId: PaneId; documentPath: string }
  | { paneType: 'intelligence_hub'; paneId: PaneId }
  | { paneType: 'knowledge_rail'; paneId: PaneId };
```

The first layout implementation should stay pragmatic. A fixed grid or a small set of layouts is preferable to building a full window manager early.

A project-tree `Pin to reference pane` action should create or reuse a `ReferenceEditor` pane with the chosen `documentPath`.

#### Intelligence hub

The intelligence hub should absorb the current `ModeAwareChatPanel` responsibilities and expand them:

- mode switcher: `Analysis`, `Editing`, `Ideation`
- current target display
- task input and run controls
- task output rendering
- context inspector for the exact `ContextBundle`
- links back to editor spans
- thread-local anchor list

The hub should not own editor selection state. It reads `activeTaskTarget` and sends task requests through the shared request builder and Tauri wrapper.

#### Draft review layer

Editing mode returns `DraftChange` objects. These should eventually be rendered non-destructively in CodeMirror with decorations:

- use CodeMirror 6 `Decoration.mark` to style the original target range
- use `Decoration.widget` to insert a review card near the target range
- show original text as dimmed or struck through
- show proposed text as a highlighted suggestion block
- provide `Accept` and `Reject` actions in the widget

Draft lifecycle:

1. `TaskResult` returns a `DraftChange`
2. frontend matches the draft target to an open editor document
3. editor enters a visible review state for that draft
4. `Reject` clears the session draft overlay
5. `Accept` is deferred to Phase 4, where a backend command applies the mutation and refreshes document state

Scope boundary:

- Phase 2 can display draft suggestions
- Phase 3 can add context and memory that improve suggestions
- Phase 4 owns file mutation, accept/reject persistence, and revalidation after edits

#### Knowledge rail

The knowledge rail should make project context an explicit control surface.

Initial rail contents:

- guide sources: prose guideline, style guide, critique rubric, rewrite guide, custom guide
- reference sources: story summary, world summary, character bible, timeline, terminology, research, custom reference
- notes: explicit-only scratch notes and loose ideation files

Recommended state:

```ts
type ActiveContextState = {
  activeContextPaths: Set<string>;
};
```

Task requests should pass active context paths as `explicitly_selected_source_paths`. The backend should still apply mode-specific policy, activation policy, and review-state gates rather than blindly including every frontend toggle.

Fact and summary review belongs in Phase 3 because it depends on persisted `ReviewableFact` and `ReviewableSummary` records.

#### Transient anchor system

Anchors should connect chat messages, task outputs, and editor spans within the active thread.

Chat to editor:

- model/task output contains target anchors
- rendered output exposes a clickable link or chip
- clicking updates `activeTaskTarget`
- focused editor pane scrolls to the matching span or selection range

Editor to chat:

- editor gutter markers represent spans referenced in the active `ChatThread`
- clicking a marker scrolls the intelligence hub to the relevant message or task output

Transience rules:

- anchors are session/thread-local until durable span IDs exist
- accepting a draft change should mark related anchors stale
- stale anchors should be dimmed and should require revalidation before reuse as task targets

#### Frontend contract rules

- convert selection state through one `SelectionTarget` adapter path
- send task execution through one Tauri wrapper path
- pass user-selected context paths explicitly
- never mutate manuscript files from Phase 2 UI code
- keep provider and Rig-specific details out of frontend task state

### Backend architecture

- Implement a Rust workspace with these subsystems:
  - `core`: shared domain types and span anchoring
  - `store`: SQLite persistence and migrations
  - `index`: parsing, chunking, entity/fact extraction, summary generation, staleness tracking
  - `retrieval`: metadata + lexical + entity/fact + vector retrieval and context packing
  - `llm`: provider abstraction and Rig-backed adapters
  - `orchestrator`: task execution, validation, job state, and draft emission
  - `desktop`: Tauri commands and filesystem integration
- Keep Rig behind internal traits; domain logic must not depend on Rig types directly.
- Support two first-class auth/provider paths:
  - official API-key providers
  - experimental subscription-auth bridge adapters, clearly isolated and marked unstable/personal-use only
- Keep the architecture open for future local-model providers, but do not make them a v1 implementation dependency.

### Core data model and interfaces

- Define stable persisted types:
  - `ProjectConfig`: root path, provider settings, guide settings, directory-role mappings
  - `ProjectDirectoryRole`: `primary_manuscript`, `reference`, `notes`, `ignore`
  - `ProjectDirectoryMapping`: normalized safe project-relative directory path, assigned role, enabled state
  - `ProjectImportCandidate`: discovered directory path, suggested role, reason for suggestion
  - `ContextSource`: source document path, source kind, activation policy, review state
  - `ContextSourceKind`: tagged source taxonomy with `guide`, `reference`, and `note`
  - `GuideKind`: `prose`, `style`, `critique`, `rewrite`, `custom`
  - `ReferenceKind`: `story_summary`, `world_summary`, `character_bible`, `timeline`, `terminology`, `research`, `custom`
  - `DocumentRecord`: path, type, title, modified time, content hash
  - `SpanRecord`: stable span ID, document ID, span type, ordinal, parent span, source range, text hash
  - `ChatThread`: mode, scope attachment, selected guides, created context policy
  - `DraftChange`: target span/region, original text, proposed text, diff, source task, state
  - `ReviewableFact`: subject, predicate, object/value, source span, review state, stale state, optional confidence once extraction produces a defensible score
  - `ReviewableSummary`: scope, text, source spans, review state, stale state, optional confidence once extraction produces a defensible score
  - `TaskRequest`: mode, task type, scope, allowed context sources, provider, guide set
  - `TaskResult`: comments, suggestions, extracted records, citations, confidence, validator outcome
  - `ContextBundle`: exact spans, guides, approved facts, approved summaries, semantic fallbacks used
- Keep output contracts structured and versioned for:
  - analysis comments/findings
  - editing diffs
  - ideation options/cards
  - entity extraction
  - fact extraction
  - summary generation
  - validator outcomes

### Ingestion, indexing, and memory

- Use a config-driven import flow in v1:
  - user selects a project root
  - app scans candidate directories under that root
  - UI prompts the user to assign roles to directories before indexing begins
  - exactly one directory must be assigned as `primary_manuscript`
  - root-level Markdown projects are represented by a normalized `.` mapping
  - zero or more additional directories may be assigned as `reference` or `notes`
- Validate directory mappings before persistence:
  - reject empty mappings
  - reject absolute paths
  - reject parent-directory traversal such as `..`
  - normalize equivalent paths such as `drafts/` to `drafts`
  - reject duplicates after normalization
- Discovery uses the most specific enabled mapping when mappings overlap, so a broad `.` mapping can coexist with a more specific `lore/` reference mapping.
- Broad discovery skips hidden/app directories such as `.git` and `.writing-assist` unless a user explicitly maps one later.
- Markdown file discovery supports common Markdown extensions case-insensitively: `.md`, `.markdown`, and `.mdown`.
- Treat folder-name heuristics as suggestions only:
  - if `chapters/` exists, suggest `primary_manuscript`
  - if `world_context/` exists, suggest `reference`
  - user confirmation overrides all heuristic guesses
- Treat directory roles as broad import buckets, not final semantic truth:
  - files inside `reference` or `notes` directories can still become document-level `guide`, `reference`, or `note` context sources
  - `guide` context sources can become active prompt/rubric context
  - `reference` context sources can become story/world/character/timeline/terminology context
  - `note` context sources remain available but should not enter prompts automatically without explicit selection or retrieval
- Add a document-level context source classification task after import:
  - prose guideline and style guide documents should be first-class `guide` sources
  - story summary, world summary, character bible, timeline, terminology, and research documents should be first-class `reference` sources
  - scratch notes and loose ideation should remain `note` sources
  - ambiguous sources should remain `custom` or unclassified rather than guessed aggressively
- Parse Markdown into:
  - headings
  - paragraphs
  - sections
  - explicit scene-break spans from thematic separators such as `---`
  - first-class scene objects derived from explicit scene breaks
  - exact byte and character offsets
  - whitespace-normalized sidecar text for retrieval/comparison
  - section-boundary metadata showing file-start, heading, or scene-break boundaries
- Treat parser-emitted `ParsedSpan` records as linear spans only:
  - heading
  - paragraph
  - explicit scene-break marker
- Treat sections, scenes, and future rolling windows as target categories for task construction rather than assuming all of them are emitted as `ParsedSpan` entries.
- Keep rolling edit windows as a later task context selection concern rather than a Phase 1 parser responsibility.
- Build these indexes:
  - metadata index
  - SQLite FTS lexical index
  - embedding index
  - entity graph
  - fact graph
  - stale dependency graph
- Ingestion pipeline:
  - discover candidate directories
  - collect user directory-role mapping
  - persist import configuration
  - discover files inside configured directories
  - classify docs from configured directory roles
  - parse spans
  - extract entities
  - extract candidate facts from reference docs
  - generate candidate summaries
  - store all derived memory as pending review
  - build cross-links from spans to entities and approved memory
- Reuse only approved summaries and approved facts in retrieval and consistency tasks.
- Mark derived memory stale whenever source hashes change.

### Retrieval and hallucination controls

- Make all model calls span-bounded and task-shaped.
- Assemble context in this order:
  - target span
  - nearby spans
  - pinned/selected guides, such as prose guideline or critique rubric
  - approved reference summaries
  - approved reference facts
  - semantic neighbors only if needed
- Use different context-source policies by mode:
  - `Analysis`: include selected guide/rubric plus relevant approved references; do not mutate text
  - `Editing`: include selected prose/style/rewrite guides and only references needed to preserve meaning/canon
  - `Ideation`: include selected guide plus broader approved references when the user asks for expansion
- Apply context-source defaults through both semantic kind and state:
  - `Pinned` and `Retrieved` sources can be included by default when mode policy allows them
  - `ExplicitOnly` sources require explicit user selection
  - `UserAuthored` and `Approved` sources can be included by default
  - `PendingReview` and `Stale` sources are excluded by default
- Prefer lexical/entity/fact retrieval over vector retrieval for names, canon, terminology, and continuity checks.
- Require evidence-first outputs for critique and fact extraction.
- Run validator tasks for rewrite suggestions before they enter the draft layer when feasible.
- Allow explicit uncertainty states such as `needs broader context` and `cannot determine`.

### Phase 3 retrieval and memory scope

Phase 3 introduces reviewable project memory and retrieval, but it should stay deterministic and review-gated before provider-generated extraction is added.

Execution order:

- define reviewable memory contracts in `core`
- add deterministic entity extraction from parsed Markdown spans
- persist reviewable memory and review/stale states in SQLite
- add context-source classification and frontend knowledge-rail state
- add deterministic fact/summary candidate scaffolding
- add memory review UI and Tauri/store commands
- add retrieval v1 and context inspector

#### Reviewable memory contracts

Core memory types should be provider-agnostic:

- `EntityCandidate`
- `ReviewableFact`
- `ReviewableSummary`
- `MemoryReviewState`
- `MemoryStalenessState`
- source document/span references

Memory reuse rules:

- pending memory is not reusable in task context
- rejected memory is not reusable in task context
- stale memory is not reusable in task context
- approved memory remains source-linked
- IDs are stable fields, not inferred from display text

#### Deterministic extraction first

The first extraction slice should not use LLM calls. It should produce conservative pending candidates only:

- repeated proper nouns
- capitalized names/phrases with noise controls
- glossary-like or structured reference lines where present
- document path and span anchors on every candidate

Fact and summary candidate generation should start as scaffolding:

- facts only from structured reference-like lines, not broad prose claims
- summaries at document/section scope as pending records
- no automatic approval
- no provider-generated summaries until the review and persistence path is stable

#### Persistence and staleness

SQLite should store:

- entity candidates
- reviewable facts
- reviewable summaries
- source references
- review states
- stale states

Store APIs should support:

- saving pending candidates
- listing pending/approved/stale records
- approving and rejecting records
- marking dependent records stale when source hashes change
- querying only reusable memory for task context

#### Knowledge rail and context-source state

The knowledge rail should be the explicit frontend control surface for source selection:

- guide toggles
- reference toggles
- note toggles
- active context path state

Task requests should pass active paths as `explicitly_selected_source_paths`. The backend must still apply task context selection policy, activation policy, and review-state gates. Frontend toggles are requests for inclusion, not authority to bypass memory review.

#### Retrieval v1

Retrieval should prefer deterministic, inspectable signals before vector fallback:

- metadata retrieval
- lexical retrieval over parser-normalized text
- approved entity/fact/summary retrieval
- vector retrieval abstraction only, with concrete embeddings deferred if needed

Retrieval tests should verify:

- pending/rejected/stale memory is excluded
- lexical/entity/fact matches rank ahead of vector fallbacks for names, canon, and terminology
- context inspector shows the exact selected context for a task

#### Context inspector and anchors

The context inspector should show:

- included context sources
- excluded context sources and exclusion reasons where available
- approved memory records used
- target spans and source anchors

Thread/editor anchors remain session-local until durable span IDs or revalidated anchors exist. Phase 4 will own stale-anchor updates after accepted draft mutations.

### Implementation phases

- Phase 0: workspace setup
  - Tauri + SvelteKit shell
  - Rust workspace structure
  - SQLite setup and migrations
  - shared types for projects, documents, spans, threads, and draft changes
- Phase 1: project ingestion and editor
  - open project folder
  - scan candidate directories
  - prompt the user to assign directory roles
  - persist project import configuration
  - discover and classify Markdown files from the configured mapping
  - parse spans
  - render file tree and CodeMirror editor
  - support document loading and selection-to-span targeting
- Phase 1.9: project context source planning
  - define the distinction between broad directory roles and document-level context source types
  - model prose/style/critique/rewrite guides as first-class selected context inputs
  - model story/world/character/timeline/terminology bibles as first-class approved reference inputs
  - define how each context source type is allowed into Analysis, Editing, and Ideation requests
- Phase 1.10: Phase 1 hardening
  - enforce safe normalized directory mappings
  - support root-level manuscript files through a `.` mapping
  - skip hidden/app directories during broad discovery
  - preserve saved mappings not present in a candidate rescan
  - clarify emitted parser span types versus task target categories
  - lift editor selection target state for Phase 2 chat/task handoff
  - gate default context-source inclusion by activation and review state
- Phase 2: mode-aware chat and task orchestration
  - add Analysis/Editing/Ideation chat modes
  - implement `SelectionTarget`, `TaskRequest`, `TaskResult`, and `ContextBundle`
  - model `SelectionTarget` across spans, sections, scenes, and future windows
  - use context source taxonomy from Phase 1.9 plus activation/review gates from Phase 1.10 in task context policies
  - connect chat UI to orchestrator
  - emit comments, idea cards, and draft changes by mode
  - define the Writer's IDE frontend workspace architecture before locking in the chat-panel layout
- Phase 3: reviewable memory and retrieval
  - define reviewable memory contracts
  - add deterministic entity extraction
  - persist memory review and stale states
  - add context-source classification and knowledge rail state
  - add deterministic fact and summary candidate scaffolding
  - add memory review UI and approval workflow
  - add retrieval v1 and context inspector
  - keep vector retrieval optional behind an abstraction
- Phase 4: diffs, validation, and consistency
  - diff review and accept/reject flow
  - render draft changes as CodeMirror review overlays
  - apply accepted changes back to Markdown files
  - mark affected anchors stale after accepted changes
  - validator task for rewrites
  - canon/terminology consistency checks using approved memory
- Phase 5: provider polish
  - API-key provider setup
  - experimental subscription-auth bridge adapter
  - provider selection/settings UX
  - logging and recovery for auth/provider failures

## Test Plan

- Import a sample project with arbitrary directory names and verify the app does not assume folder names as truth.
- Import a sample project with root-level Markdown files and verify the `.` mapping discovers them safely.
- Verify unsafe mapping paths such as absolute paths and `..` are rejected before persistence/discovery.
- Verify overlapping mappings use the most specific mapping for classification.
- Verify hidden/app directories are skipped during broad discovery from `.`.
- Verify the import flow requires one `primary_manuscript` directory before indexing can proceed.
- Verify heuristic suggestions only prefill the import UI and do not override user choices.
- Verify saved mappings are preserved if a rescan does not currently return the same candidate directory.
- Verify configured directory-role mappings persist and are reused on subsequent project opens.
- Verify file discovery only indexes Markdown files from user-enabled mapped directories.
- Verify guide/reference/note context source distinctions are preserved in task request construction.
- Verify context sources excluded by review state or activation policy do not enter context bundles by default.
- Verify CodeMirror selection offsets map to parsed spans before task construction.
- Verify the selected document target is available to the parent workspace/chat boundary, not only inside the editor component.
- Confirm paragraph/window anchors survive ordinary edits and remain stable enough for comments, chat attachment, and draft application.
- Verify Analysis mode can discuss a selected paragraph and produce findings/comments without creating draft changes.
- Verify Editing mode can produce bounded diffs only for the selected span/window and route them into the draft layer.
- Verify Ideation mode produces constrained options tied to the current selection/window and approved canon, without direct manuscript mutation.
- Confirm accepted draft changes update Markdown source files correctly and rejected changes leave source files untouched.
- Confirm extracted facts and summaries stay unavailable to retrieval until reviewed and approved.
- Modify source text after approval and verify dependent facts/summaries become stale and are excluded until refreshed and re-reviewed.
- Verify context inspector shows the exact context bundle used for each model call.
- Test both provider paths:
  - API-key path works end to end
  - bridge adapter failures are isolated and do not corrupt drafts, indexes, or project state
- Verify retrieval prefers local text, guides, approved facts, and lexical/entity matches before vector fallbacks.
- Verify consistency checks only use approved canon memory and never silently pull from pending/stale records.

## Assumptions

- v1 is a personal desktop tool, not a multi-user or commercial SaaS product.
- Markdown files remain the canonical manuscript format.
- `SvelteKit + CodeMirror` is fixed for v1; future editor replacement is enabled through abstraction, not implemented now.
- Chat is a primary UX surface, but it is mode-aware and grounded rather than a generic unconstrained assistant.
- Both API keys and experimental subscription-auth bridges are visible as equal setup choices in the UI, but bridge adapters are isolated and documented as unstable.
- All machine-derived summaries and canon facts require explicit review before reuse.
- Local models are a future extension, not part of the first implementation wave.
- Folder names are not treated as durable semantics; directory roles come from project configuration, with heuristics used only to suggest defaults during import.
