import type {
  PersistedProjectDirectoryMapping,
  ProjectDirectoryMappingDraft,
  ProjectImportCandidate
} from '$lib/project-import/types';

export function toPersistedMappings(
  mappings: ProjectDirectoryMappingDraft[]
): PersistedProjectDirectoryMapping[] {
  return mappings
    .filter((mapping): mapping is PersistedProjectDirectoryMapping => mapping.role !== null)
    .map((mapping) => ({
      path: mapping.path,
      role: mapping.role,
      enabled: mapping.enabled
    }));
}

export function applyPersistedMappingsToCandidates(
  candidates: ProjectImportCandidate[],
  persistedMappings: PersistedProjectDirectoryMapping[]
): ProjectDirectoryMappingDraft[] {
  const persistedByPath = new Map(
    persistedMappings.map((mapping) => [mapping.path, mapping] as const)
  );

  // Saved mappings override heuristic suggestions, while unmapped candidates keep their scan-time defaults.
  const mappedCandidates = candidates.map((candidate) => {
    const persistedMapping = persistedByPath.get(candidate.path);

    if (persistedMapping) {
      return {
        path: persistedMapping.path,
        role: persistedMapping.role,
        enabled: persistedMapping.enabled
      };
    }

    return {
      path: candidate.path,
      role: candidate.suggested_role,
      enabled: candidate.suggested_role !== 'ignore'
    };
  });

  const candidatePaths = new Set(candidates.map((candidate) => candidate.path));
  const missingPersistedMappings = persistedMappings
    .filter((mapping) => !candidatePaths.has(mapping.path))
    .map((mapping) => ({
      path: mapping.path,
      role: mapping.role,
      enabled: mapping.enabled
    }))
    .sort((left, right) => left.path.localeCompare(right.path));

  return [...mappedCandidates, ...missingPersistedMappings];
}
