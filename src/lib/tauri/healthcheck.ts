declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

export type HealthcheckStatus = {
  runtime: 'browser' | 'tauri';
  state: 'idle' | 'loading' | 'ready' | 'error';
  message: string;
};

export function isTauriRuntime(): boolean {
  return typeof window !== 'undefined' && typeof window.__TAURI_INTERNALS__ !== 'undefined';
}

export async function runHealthcheck(): Promise<HealthcheckStatus> {
  if (!isTauriRuntime()) {
    return {
      runtime: 'browser',
      state: 'ready',
      message: 'Browser mode detected. Tauri commands are unavailable until the desktop shell is running.'
    };
  }

  try {
    const { invoke } = await import('@tauri-apps/api/core');
    const response = await invoke<string>('healthcheck');

    return {
      runtime: 'tauri',
      state: 'ready',
      message: `Tauri healthcheck returned: ${response}`
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown Tauri invocation failure';

    return {
      runtime: 'tauri',
      state: 'error',
      message
    };
  }
}
