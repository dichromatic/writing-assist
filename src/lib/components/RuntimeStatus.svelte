<script lang="ts">
  import { onMount } from 'svelte';
  import { runHealthcheck, type HealthcheckStatus } from '$lib/tauri/healthcheck';

  let status: HealthcheckStatus = {
    runtime: 'browser',
    state: 'idle',
    message: 'Healthcheck has not run yet.'
  };

  async function refreshStatus() {
    status = {
      ...status,
      state: 'loading',
      message: 'Checking runtime bridge...'
    };

    status = await runHealthcheck();
  }

  onMount(async () => {
    await refreshStatus();
  });
</script>

<section class="panel">
  <div class="header">
    <div>
      <p class="label">Runtime bridge</p>
      <h2>Tauri healthcheck</h2>
    </div>
    <button type="button" on:click={refreshStatus}>Refresh</button>
  </div>

  <dl class="meta">
    <div>
      <dt>Runtime</dt>
      <dd>{status.runtime}</dd>
    </div>
    <div>
      <dt>State</dt>
      <dd data-state={status.state}>{status.state}</dd>
    </div>
  </dl>

  <p class="message">{status.message}</p>
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
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .label {
    margin: 0 0 0.35rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
  }

  h2 {
    margin: 0;
    font-size: 1.1rem;
  }

  button {
    border: 1px solid var(--panel-border);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--text);
    padding: 0.65rem 1rem;
    cursor: pointer;
  }

  .meta {
    display: flex;
    gap: 1.25rem;
    margin: 0 0 1rem;
  }

  .meta div {
    min-width: 8rem;
  }

  dt {
    margin-bottom: 0.25rem;
    color: var(--muted);
    font-size: 0.85rem;
  }

  dd {
    margin: 0;
    text-transform: capitalize;
  }

  [data-state='ready'] {
    color: #9fdb9a;
  }

  [data-state='error'] {
    color: #ff8b8b;
  }

  [data-state='loading'] {
    color: #ffd27f;
  }

  .message {
    margin: 0;
    color: var(--muted);
    line-height: 1.6;
  }
</style>
