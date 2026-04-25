import type { ProjectDirectoryMappingDraft } from '$lib/project-import/types';

export type ImportSelectionValidation = {
  isValid: boolean;
  primaryManuscriptCount: number;
  message: string | null;
};

export function validateImportSelection(
  mappings: ProjectDirectoryMappingDraft[]
): ImportSelectionValidation {
  // Phase 1 import is only valid when there is a single manuscript root for later discovery/parsing.
  const primaryManuscriptCount = mappings.filter(
    (mapping) => mapping.enabled && mapping.role === 'primary_manuscript'
  ).length;

  if (primaryManuscriptCount === 0) {
    return {
      isValid: false,
      primaryManuscriptCount,
      message: 'Select one primary manuscript directory before continuing.'
    };
  }

  if (primaryManuscriptCount > 1) {
    return {
      isValid: false,
      primaryManuscriptCount,
      message: 'Only one directory can be marked as the primary manuscript.'
    };
  }

  return {
    isValid: true,
    primaryManuscriptCount,
    message: null
  };
}
