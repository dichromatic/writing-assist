import { describe, expect, it } from 'vitest';

import { runDeterministicTask } from './taskExecution';
import type { DeterministicTaskCommandRequest } from '$lib/task/types';

function request(): DeterministicTaskCommandRequest {
  return {
    mode: 'analysis',
    task_type: 'analyze_selection',
    target: {
      document_path: 'chapters/chapter-1.md',
      selected_text: 'Selected paragraph.',
      start_char: 10,
      end_char: 29,
      anchors: [{ kind: 'span', ordinal: 2 }]
    },
    available_sources: [],
    explicitly_selected_source_paths: []
  };
}

describe('runDeterministicTask', () => {
  it('returns an explicit browser fallback instead of pretending to call Tauri', async () => {
    await expect(runDeterministicTask(request())).resolves.toEqual({
      runtime: 'browser',
      state: 'unavailable',
      message:
        'Browser mode detected. Deterministic task execution requires the Tauri desktop runtime.',
      result: null
    });
  });
});
