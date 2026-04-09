<script lang="ts">
  import ModeCard from '$lib/components/ModeCard.svelte';
  import RuntimeStatus from '$lib/components/RuntimeStatus.svelte';

  const modes = [
    {
      title: 'Analysis',
      description: 'Critique selected text, inspect consistency, and discuss style or structure without mutating the manuscript.'
    },
    {
      title: 'Editing',
      description: 'Generate bounded draft changes for paragraphs and windows, with diffs routed through explicit review.'
    },
    {
      title: 'Ideation',
      description: 'Brainstorm within current context and approved canon, then promote useful output into notes or edits later.'
    }
  ];
</script>

<svelte:head>
  <title>Writing Assist</title>
  <meta
    name="description"
    content="Local-first writing workspace for analysis, editing, and ideation."
  />
</svelte:head>

<div class="page">
  <section class="hero">
    <p class="eyebrow">Phase 0 scaffold</p>
    <h1>Writing Assist</h1>
    <p class="summary">
      Local-first writing workspace for Markdown projects. Frontend is SvelteKit + CodeMirror;
      backend is Rust + Tauri + SQLite.
    </p>
  </section>

  <section class="grid">
    {#each modes as mode}
      <ModeCard {...mode} />
    {/each}
  </section>

  <RuntimeStatus />
</div>

<style>
  .page {
    max-width: 1100px;
    margin: 0 auto;
    padding: 4rem 1.5rem;
  }

  .hero {
    margin-bottom: 2rem;
  }

  .eyebrow {
    margin: 0 0 0.75rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.78rem;
  }

  h1 {
    margin: 0;
    font-size: clamp(2.5rem, 8vw, 5rem);
    line-height: 0.95;
  }

  .summary {
    max-width: 48rem;
    margin: 1rem 0 0;
    color: var(--muted);
    font-size: 1.1rem;
    line-height: 1.7;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
  }
</style>
