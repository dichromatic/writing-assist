# Writing Assist v1 Implementation Plan

## Summary

Build a personal, local-first desktop writing workspace for Markdown projects using `Tauri + SvelteKit + CodeMirror + Rust + SQLite + Rig`. The app combines a primary chat/discussion surface with structured editorial tools, organized into three first-class modes: `Analysis`, `Editing`, and `Ideation`.

The manuscript folder remains canonical, but the app uses an internal draft layer for AI suggestions and human review. Project import is configuration-driven: the user chooses which directories contain primary manuscript content and which directories provide supporting reference or notes material. All model work is span-bounded, retrieval is hybrid and source-linked, and all machine-derived memory is review-gated before reuse.

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
- Preserve structured non-chat actions as shortcuts into the same backend pass system:
  - comment on selection
  - rewrite selection
  - run pass on scene
  - check consistency

### Frontend architecture

- Use `SvelteKit` as the frontend framework inside a Tauri desktop shell.
- Use `CodeMirror` as the v1 editor because Markdown files are the primary source format and paragraph/window diffs map naturally to text editing.
- Keep the editor integration behind an internal editor abstraction so a later `ProseMirror`/`Tiptap` frontend can replace the editing surface without changing backend contracts.
- Implement frontend modules for:
  - project/file navigation
  - Markdown editor with character selection mapped to parsed spans
  - document workspace for loaded Markdown documents
  - chat panel with mode-aware controls
  - comments rail and jump-to-span behavior
  - diff review and accept/reject flow
  - context inspector showing exact retrieved context
  - memory review UI for facts and summaries

### Backend architecture

- Implement a Rust workspace with these subsystems:
  - `core`: shared domain types and span anchoring
  - `store`: SQLite persistence and migrations
  - `index`: parsing, chunking, entity/fact extraction, summary generation, staleness tracking
  - `retrieval`: metadata + lexical + entity/fact + vector retrieval and context packing
  - `llm`: provider abstraction and Rig-backed adapters
  - `orchestrator`: pass execution, validation, job state, and draft emission
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
  - `DraftChange`: target span/region, original text, proposed text, diff, source pass, state
  - `ReviewableFact`: subject, predicate, object/value, source span, confidence, review state, stale state
  - `ReviewableSummary`: scope, text, source spans, confidence, review state, stale state
  - `PassRequest`: mode, pass type, scope, allowed context sources, provider, guide set
  - `PassResult`: comments, suggestions, extracted records, citations, confidence, validator outcome
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
- Add a document-level context source classification pass after import:
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
- Treat sections, scenes, and future rolling windows as target categories for pass construction rather than assuming all of them are emitted as `ParsedSpan` entries.
- Keep rolling edit windows as a later context-assembly concern rather than a Phase 1 parser responsibility.
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
- Reuse only approved summaries and approved facts in retrieval and consistency passes.
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
- Run validator passes for rewrite suggestions before they enter the draft layer when feasible.
- Allow explicit uncertainty states such as `needs broader context` and `cannot determine`.

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
  - clarify emitted parser span types versus pass target categories
  - lift editor selection target state for Phase 2 chat/pass handoff
  - gate default context-source inclusion by activation and review state
- Phase 2: mode-aware chat and pass orchestration
  - add Analysis/Editing/Ideation chat modes
  - implement `SelectionTarget`, `PassRequest`, `PassResult`, and `ContextBundle`
  - model `SelectionTarget` across spans, sections, scenes, and future windows
  - use context source taxonomy from Phase 1.9 plus activation/review gates from Phase 1.10 in pass context policies
  - connect chat UI to orchestrator
  - emit comments, idea cards, and draft changes by mode
- Phase 3: reviewable memory and retrieval
  - add entities, candidate facts, candidate summaries
  - add memory review UI and approval workflow
  - add hybrid retrieval and context inspector
  - add stale tracking
- Phase 4: diffs, validation, and consistency
  - diff review and accept/reject flow
  - apply accepted changes back to Markdown files
  - validator pass for rewrites
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
- Verify guide/reference/note context source distinctions are preserved in pass request construction.
- Verify context sources excluded by review state or activation policy do not enter context bundles by default.
- Verify CodeMirror selection offsets map to parsed spans before pass construction.
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
