<script lang="ts">
  export let title: string;
  export let description: string;
</script>

<article class="card">
  <h3>{title}</h3>
  <p>{description}</p>
</article>

<style>
  .card {
    padding: 1rem;
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01));
  }

  h3 {
    margin: 0 0 0.5rem;
    font-size: 1.05rem;
  }

  p {
    margin: 0;
    color: var(--muted);
    line-height: 1.5;
  }
</style>
