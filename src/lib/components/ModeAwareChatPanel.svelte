<script lang="ts">
  import {
    createKnowledgeRailState,
    setKnowledgeRailActivePaths,
    toggleKnowledgeRailPath
  } from '$lib/context-sources/knowledgeRail';
  import type { DocumentSelectionTarget } from '$lib/project-import/selection';
  import { buildDeterministicTaskRequest } from '$lib/task/requestBuilder';
  import type { ContextSource, ConversationMode, TaskOutput } from '$lib/task/types';
  import {
    runDeterministicTask,
    type TaskExecutionStatus
  } from '$lib/tauri/taskExecution';

  let {
    selectionTarget,
    availableSources = []
  }: {
    selectionTarget: DocumentSelectionTarget | null;
    availableSources?: ContextSource[];
  } = $props();

  const modeOptions: Array<{
    mode: ConversationMode;
    label: string;
    description: string;
  }> = [
    {
      mode: 'analysis',
      label: 'Analysis',
      description: 'Critique or explain the selected text without proposing manuscript edits.'
    },
    {
      mode: 'editing',
      label: 'Editing',
      description: 'Produce bounded draft suggestions for review, not direct file changes.'
    },
    {
      mode: 'ideation',
      label: 'Ideation',
      description: 'Generate constrained options tied to the current selection.'
    }
  ];

  let activeMode = $state<ConversationMode>('analysis');
  let execution = $state<TaskExecutionStatus | null>(null);
  let isRunning = $state(false);
  let buildError = $state<string | null>(null);
  let activeContextPaths = $state<string[]>([]);

  let canRun = $derived(Boolean(selectionTarget?.selectedText) && !isRunning);
  let knowledgeRail = $derived(
    setKnowledgeRailActivePaths(createKnowledgeRailState(availableSources), activeContextPaths)
  );

  function outputTitle(output: TaskOutput): string {
    switch (output.output_type) {
      case 'analysis_comment':
        return 'Analysis comment';
      case 'draft_change':
        return 'Draft suggestion';
      case 'idea_card':
        return 'Idea card';
    }
  }

  function contextSourceLabel(source: ContextSource): string {
    switch (source.kind.source_type) {
      case 'guide':
        return `${source.kind.source_kind.replaceAll('_', ' ')} guide`;
      case 'reference':
        return source.kind.source_kind.replaceAll('_', ' ');
      case 'note':
        return 'note';
    }
  }

  async function runTask() {
    if (!selectionTarget) {
      buildError = 'Select text before running a task.';
      execution = null;
      return;
    }

    const requestResult = buildDeterministicTaskRequest(activeMode, selectionTarget, knowledgeRail);

    if (!requestResult.ok) {
      buildError = requestResult.message;
      execution = null;
      return;
    }

    buildError = null;
    isRunning = true;

    try {
      // Phase 2.7 only displays structured outputs; it does not apply edits to the manuscript.
      execution = await runDeterministicTask(requestResult.request);
    } finally {
      isRunning = false;
    }
  }
</script>

<section class="chat-panel" data-mode={activeMode}>
  <div class="panel-header">
    <div>
      <p class="label">Phase 2.7</p>
      <h3>Mode-aware chat</h3>
    </div>
    <button type="button" disabled={!canRun} onclick={runTask}>
      {isRunning ? 'Running...' : 'Run deterministic task'}
    </button>
  </div>

  <div class="mode-switcher" aria-label="Conversation mode">
    {#each modeOptions as option (option.mode)}
      <button
        type="button"
        class:selected={activeMode === option.mode}
        aria-pressed={activeMode === option.mode}
        onclick={() => (activeMode = option.mode)}
      >
        <strong>{option.label}</strong>
        <span>{option.description}</span>
      </button>
    {/each}
  </div>

  <div class="scope-card">
    <p class="label">Current task target</p>
    {#if selectionTarget?.selectedText}
      <p>
        {selectionTarget.documentPath}; chars {selectionTarget.startChar}-{selectionTarget.endChar}; spans
        {selectionTarget.overlappingSpanOrdinals.join(', ') || 'none'}.
      </p>
      <blockquote>{selectionTarget.selectedText}</blockquote>
    {:else}
      <p>Select manuscript text in the editor before running Analysis, Editing, or Ideation.</p>
    {/if}
  </div>

  <div class="context-card">
    <p class="label">Selected context sources</p>
    {#if knowledgeRail.availableSources.length > 0}
      <p class="context-note">
        Project sources stay opt-in at this stage. Toggle only the files you want attached to the task.
      </p>
      <div class="context-list">
        {#each knowledgeRail.availableSources as source (source.path)}
          <label class="context-source">
            <input
              type="checkbox"
              checked={knowledgeRail.activeContextPaths.includes(source.path)}
              onchange={() => {
                activeContextPaths = toggleKnowledgeRailPath(knowledgeRail, source.path).activeContextPaths;
              }}
            />
            <span>
              <strong>{contextSourceLabel(source)}</strong>
              <small>{source.path}</small>
            </span>
          </label>
        {/each}
      </div>
    {:else}
      <p>No guide, reference, or note sources are currently available for this project.</p>
    {/if}
  </div>

  {#if buildError}
    <p class="message" data-state="error">{buildError}</p>
  {/if}

  {#if execution}
    <p class="message" data-state={execution.state}>{execution.message}</p>
  {/if}

  {#if execution?.result}
    <div class="output-list">
      {#each execution.result.outputs as output, index (`${output.output_type}-${index}`)}
        <article class="task-output">
          <p class="label">{outputTitle(output)} {index + 1}</p>

          {#if output.output_type === 'analysis_comment'}
            <p>{output.output.message}</p>
          {:else if output.output_type === 'draft_change'}
            <p class="draft-note">Suggestion only. This does not mutate the manuscript file.</p>
            <div class="draft-grid">
              <div>
                <strong>Original</strong>
                <blockquote>{output.output.original_text}</blockquote>
              </div>
              <div>
                <strong>Proposed</strong>
                <blockquote>{output.output.proposed_text}</blockquote>
              </div>
            </div>
          {:else if output.output_type === 'idea_card'}
            <h4>{output.output.title}</h4>
            <p>{output.output.body}</p>
          {/if}
        </article>
      {/each}
    </div>
  {/if}
</section>

<style>
  .chat-panel {
    display: grid;
    gap: 1rem;
    margin-top: 1rem;
    padding: 1rem;
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.015));
  }

  .chat-panel[data-mode='analysis'] {
    border-color: rgba(132, 185, 255, 0.36);
  }

  .chat-panel[data-mode='editing'] {
    border-color: rgba(255, 198, 123, 0.42);
  }

  .chat-panel[data-mode='ideation'] {
    border-color: rgba(142, 221, 180, 0.38);
  }

  .panel-header {
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

  h3,
  h4 {
    margin: 0;
  }

  button {
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--text);
    padding: 0.75rem 0.9rem;
    cursor: pointer;
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.58;
  }

  .mode-switcher {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
  }

  .mode-switcher button {
    display: grid;
    gap: 0.35rem;
    text-align: left;
  }

  .mode-switcher button.selected {
    border-color: var(--accent);
    background: rgba(255, 255, 255, 0.08);
  }

  .mode-switcher span,
  .scope-card p,
  .context-card p,
  .message,
  .draft-note {
    color: var(--muted);
  }

  .scope-card,
  .context-card,
  .task-output {
    padding: 0.9rem 1rem;
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.03);
  }

  .scope-card p,
  .context-card p,
  .task-output p,
  .draft-note,
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

  .message {
    margin: 0;
  }

  [data-state='error'] {
    color: #ff8b8b;
  }

  [data-state='unavailable'] {
    color: #ffc67b;
  }

  .output-list {
    display: grid;
    gap: 0.75rem;
  }

  .context-note {
    margin-bottom: 0.75rem;
  }

  .context-list {
    display: grid;
    gap: 0.65rem;
  }

  .context-source {
    display: flex;
    gap: 0.75rem;
    align-items: start;
    padding: 0.75rem;
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.02);
  }

  .context-source span {
    display: grid;
    gap: 0.15rem;
  }

  .context-source small {
    color: var(--muted);
    word-break: break-word;
  }

  .draft-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 0.75rem;
  }

  @media (max-width: 720px) {
    .panel-header,
    .mode-switcher,
    .draft-grid {
      grid-template-columns: 1fr;
      display: grid;
    }
  }
</style>
