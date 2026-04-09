import { describe, expect, it } from 'vitest';

import { validateImportSelection } from '$lib/project-import/validation';
import type { ProjectDirectoryMappingDraft } from '$lib/project-import/types';

function mapping(
  path: string,
  role: ProjectDirectoryMappingDraft['role'],
  enabled = true
): ProjectDirectoryMappingDraft {
  return { path, role, enabled };
}

describe('validateImportSelection', () => {
  it('requires one primary manuscript directory', () => {
    const result = validateImportSelection([
      mapping('notes', 'notes'),
      mapping('reference', 'reference')
    ]);

    expect(result).toEqual({
      isValid: false,
      primaryManuscriptCount: 0,
      message: 'Select one primary manuscript directory before continuing.'
    });
  });

  it('rejects multiple primary manuscript directories', () => {
    const result = validateImportSelection([
      mapping('drafts', 'primary_manuscript'),
      mapping('chapters', 'primary_manuscript')
    ]);

    expect(result).toEqual({
      isValid: false,
      primaryManuscriptCount: 2,
      message: 'Only one directory can be marked as the primary manuscript.'
    });
  });

  it('accepts exactly one enabled primary manuscript directory', () => {
    const result = validateImportSelection([
      mapping('drafts', 'primary_manuscript'),
      mapping('reference', 'reference'),
      mapping('archive', 'primary_manuscript', false)
    ]);

    expect(result).toEqual({
      isValid: true,
      primaryManuscriptCount: 1,
      message: null
    });
  });
});
