<script lang="ts">
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';
  import { EditorState } from '@codemirror/state';
  import { EditorView, lineNumbers } from '@codemirror/view';

  let { value = '' }: { value?: string } = $props();

  const dispatch = createEventDispatcher<{
    selectionChange: { anchorChar: number; headChar: number };
  }>();

  let editorHost: HTMLDivElement;
  let view: EditorView | null = null;

  function dispatchSelectionChange(nextView: EditorView) {
    const selection = nextView.state.selection.main;

    dispatch('selectionChange', {
      anchorChar: selection.anchor,
      headChar: selection.head
    });
  }

  function buildState(markdown: string) {
    return EditorState.create({
      doc: markdown,
      extensions: [
        lineNumbers(),
        EditorView.lineWrapping,
        EditorView.editable.of(false),
        EditorView.updateListener.of((update) => {
          if (update.selectionSet) {
            // Phase 1.8 exposes CodeMirror selection offsets so callers can map them to parsed spans.
            dispatchSelectionChange(update.view);
          }
        }),
        // Keep Phase 1.7 theming local and minimal; a proper editor theme system is a later concern.
        EditorView.theme({
          '&': {
            minHeight: '26rem',
            borderRadius: '16px',
            backgroundColor: '#11161f',
            color: '#e6edf3'
          },
          '.cm-scroller': {
            fontFamily: '"Iosevka", "JetBrains Mono", monospace',
            lineHeight: '1.65'
          },
          '.cm-gutters': {
            backgroundColor: '#0d1117',
            color: '#7d8590',
            borderRight: '1px solid rgba(255, 255, 255, 0.08)'
          },
          '.cm-activeLineGutter': {
            backgroundColor: 'rgba(255, 255, 255, 0.04)'
          },
          '.cm-content': {
            caretColor: '#e6edf3'
          }
        })
      ]
    });
  }

  onMount(() => {
    // Phase 1.7 uses CodeMirror as a read-only Markdown viewer before edit/diff workflows exist.
    view = new EditorView({
      parent: editorHost,
      state: buildState(value)
    });
    dispatchSelectionChange(view);
  });

  onDestroy(() => {
    view?.destroy();
  });

  $effect(() => {
    if (!view) {
      return;
    }

    const currentValue = view.state.doc.toString();

    if (currentValue === value) {
      return;
    }

    view.setState(buildState(value));
  });
</script>

<div class="editor-shell" bind:this={editorHost}></div>

<style>
  .editor-shell {
    overflow: hidden;
    border: 1px solid var(--panel-border);
    border-radius: 16px;
  }
</style>
