<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import MarkdownEditor from '$lib/components/MarkdownEditor.svelte';
  import {
    mapSelectionToParsedSpans,
    type DocumentSelectionTarget,
    type ParsedSelection
  } from '$lib/project-import/selection';
  import type { LoadedDocument } from '$lib/project-import/types';

  let { loadedDocument }: { loadedDocument: LoadedDocument } = $props();

  const dispatch = createEventDispatcher<{
    targetChange: DocumentSelectionTarget;
  }>();

  let selection = $state<ParsedSelection>({
    selectedText: '',
    startChar: 0,
    endChar: 0,
    overlappingSpanOrdinals: [],
    primarySpanOrdinal: null
  });
  let lastLoadedDocument: LoadedDocument | null = null;

  function publishSelectionTarget(nextSelection: ParsedSelection) {
    dispatch('targetChange', {
      ...nextSelection,
      documentPath: loadedDocument.document.path
    });
  }

  function updateSelection(event: CustomEvent<{ anchorChar: number; headChar: number }>) {
    selection = mapSelectionToParsedSpans(
      loadedDocument.parsed.spans,
      loadedDocument.markdown,
      event.detail.anchorChar,
      event.detail.headChar
    );
    // Phase 2 chat/task orchestration will consume this parent-level document target.
    publishSelectionTarget(selection);
  }

  $effect(() => {
    if (lastLoadedDocument === loadedDocument) {
      return;
    }

    lastLoadedDocument = loadedDocument;
    // Reset selection whenever the loaded document changes so Phase 2 never receives stale span targeting data.
    selection = mapSelectionToParsedSpans(
      loadedDocument.parsed.spans,
      loadedDocument.markdown,
      0,
      0
    );
    publishSelectionTarget(selection);
  });
</script>

<article class="workspace">
  <div class="workspace-header">
    <div>
      <p class="label">Phase 1.8</p>
      <h3>{loadedDocument.document.path}</h3>
    </div>
    <dl class="parsed-summary">
      <div>
        <dt>Spans</dt>
        <dd>{loadedDocument.parsed.spans.length}</dd>
      </div>
      <div>
        <dt>Sections</dt>
        <dd>{loadedDocument.parsed.sections.length}</dd>
      </div>
      <div>
        <dt>Scenes</dt>
        <dd>{loadedDocument.parsed.scenes.length}</dd>
      </div>
    </dl>
  </div>

  <MarkdownEditor value={loadedDocument.markdown} on:selectionChange={updateSelection} />

  <aside class="selection-panel">
    <div>
      <p class="label">Selection target</p>
      {#if selection.selectedText}
        <p>
          Chars {selection.startChar}-{selection.endChar}; spans {selection.overlappingSpanOrdinals.join(', ')}.
        </p>
        <blockquote>{selection.selectedText}</blockquote>
      {:else}
        <p>Select text in the document to prepare a target for Analysis, Editing, or Ideation.</p>
      {/if}
    </div>
  </aside>
</article>

<style>
  .workspace {
    display: grid;
    gap: 1rem;
    margin-top: 1rem;
    padding: 1rem;
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.03);
  }

  .workspace-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: start;
  }

  .label {
    margin: 0 0 0.35rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
  }

  h3 {
    margin: 0;
  }

  .parsed-summary {
    display: flex;
    gap: 0.75rem;
    margin: 0;
  }

  .parsed-summary div,
  .selection-panel {
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.03);
  }

  .parsed-summary div {
    min-width: 4.5rem;
    padding: 0.65rem 0.75rem;
  }

  .parsed-summary dt,
  .parsed-summary dd {
    margin: 0;
  }

  .parsed-summary dt,
  .selection-panel p {
    color: var(--muted);
  }

  .parsed-summary dt {
    font-size: 0.75rem;
  }

  .parsed-summary dd {
    margin-top: 0.2rem;
    font-weight: 700;
  }

  .selection-panel {
    padding: 0.85rem 1rem;
  }

  .selection-panel p,
  blockquote {
    margin: 0.35rem 0 0;
  }

  blockquote {
    max-height: 9rem;
    overflow: auto;
    padding-left: 0.85rem;
    border-left: 3px solid var(--accent);
    color: var(--text);
    white-space: pre-wrap;
  }

  @media (max-width: 720px) {
    .workspace-header {
      display: grid;
      grid-template-columns: 1fr;
    }
  }
</style>
