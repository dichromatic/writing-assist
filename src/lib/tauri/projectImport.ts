import { isTauriRuntime } from '$lib/tauri/healthcheck';
import type { ProjectImportCandidate } from '$lib/project-import/types';

export async function scanProjectImportCandidates(root: string): Promise<ProjectImportCandidate[]> {
  if (!isTauriRuntime()) {
    throw new Error('Project import scanning requires the Tauri desktop runtime.');
  }

  // Dynamic import keeps browser-only development usable while the desktop shell is absent.
  const { invoke } = await import('@tauri-apps/api/core');

  return invoke<ProjectImportCandidate[]>('scan_project_import_candidates', { root });
}
