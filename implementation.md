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
  - Markdown editor with paragraph/window selection
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
  - `ProjectDirectoryMapping`: relative directory path, assigned role, enabled state
  - `ProjectImportCandidate`: discovered directory path, suggested role, reason for suggestion
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
  - zero or more additional directories may be assigned as `reference` or `notes`
- Treat folder-name heuristics as suggestions only:
  - if `chapters/` exists, suggest `primary_manuscript`
  - if `world_context/` exists, suggest `reference`
  - user confirmation overrides all heuristic guesses
- Parse Markdown into:
  - headings
  - paragraphs
  - sections
  - rolling edit windows of 2-4 paragraphs
  - scene spans only when explicit headings/separators make them clear
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
  - selected guide/rubric
  - approved summaries
  - approved facts
  - semantic neighbors only if needed
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
  - support selection/window targeting
- Phase 2: mode-aware chat and pass orchestration
  - add Analysis/Editing/Ideation chat modes
  - implement `PassRequest`, `PassResult`, and `ContextBundle`
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
- Verify the import flow requires one `primary_manuscript` directory before indexing can proceed.
- Verify heuristic suggestions only prefill the import UI and do not override user choices.
- Verify configured directory-role mappings persist and are reused on subsequent project opens.
- Verify file discovery only indexes Markdown files from user-enabled mapped directories.
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
