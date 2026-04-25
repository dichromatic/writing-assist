import { isTauriRuntime } from '$lib/tauri/healthcheck';
import type { DeterministicTaskCommandRequest, TaskResult } from '$lib/task/types';

export type TaskExecutionStatus =
  | {
      runtime: 'browser';
      state: 'unavailable';
      message: string;
      result: null;
    }
  | {
      runtime: 'tauri';
      state: 'ready';
      message: string;
      result: TaskResult;
    }
  | {
      runtime: 'tauri';
      state: 'error';
      message: string;
      result: null;
    };

export async function runDeterministicTask(
  request: DeterministicTaskCommandRequest
): Promise<TaskExecutionStatus> {
  if (!isTauriRuntime()) {
    return {
      runtime: 'browser',
      state: 'unavailable',
      message:
        'Browser mode detected. Deterministic task execution requires the Tauri desktop runtime.',
      result: null
    };
  }

  try {
    // Dynamic import keeps the browser demo from loading Tauri-only APIs.
    const { invoke } = await import('@tauri-apps/api/core');
    const result = await invoke<TaskResult>('run_deterministic_task_command', { request });

    return {
      runtime: 'tauri',
      state: 'ready',
      message: 'Deterministic task execution completed.',
      result
    };
  } catch (error) {
    return {
      runtime: 'tauri',
      state: 'error',
      message: error instanceof Error ? error.message : String(error),
      result: null
    };
  }
}
