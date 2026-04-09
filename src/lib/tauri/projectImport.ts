import { isTauriRuntime } from '$lib/tauri/healthcheck';
import type {
  PersistedProjectDirectoryMapping,
  ProjectConfig,
  ProjectImportCandidate
} from '$lib/project-import/types';

export async function scanProjectImportCandidates(root: string): Promise<ProjectImportCandidate[]> {
  if (!isTauriRuntime()) {
    throw new Error('Project import scanning requires the Tauri desktop runtime.');
  }

  // Dynamic import keeps browser-only development usable while the desktop shell is absent.
  const { invoke } = await import('@tauri-apps/api/core');

  return invoke<ProjectImportCandidate[]>('scan_project_import_candidates', { root });
}

export async function saveProjectImportConfiguration(
  root: string,
  mappings: PersistedProjectDirectoryMapping[]
): Promise<ProjectConfig> {
  if (!isTauriRuntime()) {
    throw new Error('Project configuration persistence requires the Tauri desktop runtime.');
  }

  const { invoke } = await import('@tauri-apps/api/core');

  return invoke<ProjectConfig>('save_project_import_configuration', {
    root,
    mappings
  });
}

export async function loadProjectImportConfiguration(root: string): Promise<ProjectConfig | null> {
  if (!isTauriRuntime()) {
    throw new Error('Project configuration loading requires the Tauri desktop runtime.');
  }

  const { invoke } = await import('@tauri-apps/api/core');

  return invoke<ProjectConfig | null>('load_project_import_configuration', { root });
}
