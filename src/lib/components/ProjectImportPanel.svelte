<script lang="ts">
  import { validateImportSelection } from '$lib/project-import/validation';
  import type {
    ProjectDirectoryMappingDraft,
    ProjectDirectoryRole,
    ProjectImportCandidate
  } from '$lib/project-import/types';
  import { scanProjectImportCandidates } from '$lib/tauri/projectImport';

  const roleOptions: Array<{ value: ProjectDirectoryRole | null; label: string }> = [
    { value: null, label: 'Unassigned' },
    { value: 'primary_manuscript', label: 'Primary manuscript' },
    { value: 'reference', label: 'Reference' },
    { value: 'notes', label: 'Notes' },
    { value: 'ignore', label: 'Ignore' }
  ];

  let root = '';
  let scanState: 'idle' | 'loading' | 'error' | 'ready' = 'idle';
  let scanMessage = 'Enter a project root and scan for candidate directories.';
  let candidates: ProjectImportCandidate[] = [];
  let mappings: ProjectDirectoryMappingDraft[] = [];
  let selectionState = validateImportSelection(mappings);

  function resetMappings(nextCandidates: ProjectImportCandidate[]) {
    // Keep the UI mapping shape close to the persisted model that Phase 1.4 will store.
    mappings = nextCandidates.map((candidate) => ({
      path: candidate.path,
      role: candidate.suggested_role,
      enabled: candidate.suggested_role !== 'ignore'
    }));
  }

  async function scan() {
    scanState = 'loading';
    scanMessage = 'Scanning candidate directories...';

    try {
      candidates = await scanProjectImportCandidates(root);
      resetMappings(candidates);
      scanState = 'ready';
      scanMessage =
        candidates.length > 0
          ? 'Review the suggested roles and assign exactly one primary manuscript directory.'
          : 'No candidate directories were found under the provided root.';
    } catch (error) {
      scanState = 'error';
      scanMessage = error instanceof Error ? error.message : 'Candidate scan failed.';
      candidates = [];
      mappings = [];
    }
  }

  function updateRole(path: string, role: string) {
    mappings = mappings.map((mapping) =>
      mapping.path === path
        ? {
            ...mapping,
            role: role === '' ? null : (role as ProjectDirectoryRole),
            enabled: role !== 'ignore'
          }
        : mapping
    );
  }

  // Keep validation derived from the editable mapping state so the UI mirrors the Phase 1 import rules.
  $: selectionState = validateImportSelection(mappings);
</script>

<section class="panel">
  <div class="header">
    <div>
      <p class="label">Phase 1.3</p>
      <h2>Project import</h2>
    </div>
  </div>

  <div class="controls">
    <label>
      <span>Project root</span>
      <input bind:value={root} placeholder="/path/to/project" />
    </label>
    <button type="button" disabled={!root || scanState === 'loading'} on:click={scan}>
      {scanState === 'loading' ? 'Scanning...' : 'Scan directories'}
    </button>
  </div>

  <p class="message" data-state={scanState}>{scanMessage}</p>

  {#if candidates.length > 0}
    <div class="candidate-list">
      {#each candidates as candidate}
        <article class="candidate">
          <div class="candidate-header">
            <div>
              <h3>{candidate.path}</h3>
              <p>
                {candidate.contains_markdown_files
                  ? 'Contains Markdown files'
                  : 'No Markdown files detected'}
              </p>
            </div>
            <label>
              <span>Role</span>
              <select
                value={mappings.find((mapping) => mapping.path === candidate.path)?.role ?? ''}
                on:change={(event) => updateRole(candidate.path, event.currentTarget.value)}
              >
                {#each roleOptions as option}
                  <option value={option.value ?? ''}>{option.label}</option>
                {/each}
              </select>
            </label>
          </div>

          {#if candidate.suggestion_reasons.length > 0}
            <p class="reasons">
              Suggested because:
              {candidate.suggestion_reasons.join(', ').replaceAll('_', ' ')}
            </p>
          {/if}
        </article>
      {/each}
    </div>

    <div class="validation" data-valid={selectionState.isValid}>
      <p>
        {selectionState.message ??
          'Import configuration is valid. Persistence and project open come in the next slices.'}
      </p>
    </div>
  {/if}
</section>

<style>
  .panel {
    margin-top: 1.5rem;
    padding: 1.25rem;
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    background: rgba(23, 27, 33, 0.78);
    backdrop-filter: blur(10px);
  }

  .header {
    margin-bottom: 1rem;
  }

  .label {
    margin: 0 0 0.35rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
  }

  h2,
  h3 {
    margin: 0;
  }

  .controls {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.75rem;
    align-items: end;
  }

  label {
    display: grid;
    gap: 0.4rem;
  }

  label span,
  .candidate p,
  .reasons,
  .message,
  .validation p {
    color: var(--muted);
  }

  input,
  select,
  button {
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--text);
    padding: 0.75rem 0.9rem;
  }

  button {
    cursor: pointer;
  }

  .message {
    margin: 0.85rem 0 0;
  }

  [data-state='error'] {
    color: #ff8b8b;
  }

  .candidate-list {
    display: grid;
    gap: 0.85rem;
    margin-top: 1rem;
  }

  .candidate {
    padding: 1rem;
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.02);
  }

  .candidate-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
  }

  .candidate p,
  .reasons,
  .validation p {
    margin: 0.45rem 0 0;
  }

  .validation {
    margin-top: 1rem;
    padding: 0.9rem 1rem;
    border-radius: 14px;
    border: 1px solid var(--panel-border);
    background: rgba(255, 255, 255, 0.03);
  }

  .validation[data-valid='true'] {
    border-color: rgba(159, 219, 154, 0.4);
  }

  @media (max-width: 720px) {
    .controls,
    .candidate-header {
      grid-template-columns: 1fr;
      display: grid;
    }
  }
</style>
