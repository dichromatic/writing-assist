import { describe, expect, it } from 'vitest';

import { applyPersistedMappingsToCandidates, toPersistedMappings } from '$lib/project-import/mappings';
import type {
  PersistedProjectDirectoryMapping,
  ProjectDirectoryMappingDraft,
  ProjectImportCandidate
} from '$lib/project-import/types';

function candidate(
  path: string,
  suggested_role: ProjectImportCandidate['suggested_role']
): ProjectImportCandidate {
  return {
    path,
    contains_markdown_files: true,
    suggested_role,
    suggestion_reasons: []
  };
}

function persisted(
  path: string,
  role: PersistedProjectDirectoryMapping['role'],
  enabled = true
): PersistedProjectDirectoryMapping {
  return { path, role, enabled };
}

function draft(
  path: string,
  role: ProjectDirectoryMappingDraft['role'],
  enabled = true
): ProjectDirectoryMappingDraft {
  return { path, role, enabled };
}

describe('toPersistedMappings', () => {
  it('drops unassigned draft mappings before persistence', () => {
    expect(
      toPersistedMappings([
        draft('chapters', 'primary_manuscript'),
        draft('notes', null),
        draft('lore', 'reference')
      ])
    ).toEqual([
      persisted('chapters', 'primary_manuscript'),
      persisted('lore', 'reference')
    ]);
  });
});

describe('applyPersistedMappingsToCandidates', () => {
  it('overrides heuristic suggestions with saved mappings on matching paths', () => {
    expect(
      applyPersistedMappingsToCandidates(
        [candidate('chapters', 'primary_manuscript'), candidate('lore', 'reference')],
        [persisted('chapters', 'notes', false)]
      )
    ).toEqual([
      draft('chapters', 'notes', false),
      draft('lore', 'reference', true)
    ]);
  });

  it('ignores saved mappings for directories not present in the current scan', () => {
    expect(
      applyPersistedMappingsToCandidates([candidate('drafts', null)], [
        persisted('archive', 'notes')
      ])
    ).toEqual([draft('drafts', null, true)]);
  });
});
