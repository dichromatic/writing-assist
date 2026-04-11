import type { ProjectDocumentEntry } from '$lib/project-import/types';
import type { ContextSource, ContextSourceKind } from '$lib/task/types';

// Keep the frontend heuristics aligned with `crates/core/src/context.rs` so both sides classify
// import-time project documents the same way before source-specific preferences are persisted.
export function classifyContextSourceKind(
  documentType: ProjectDocumentEntry['document_type'],
  path: string,
  explicitKind: ContextSourceKind | null = null
): ContextSourceKind | null {
  if (explicitKind) {
    return explicitKind;
  }

  switch (documentType) {
    case 'manuscript':
      return null;
    case 'note':
      return { source_type: 'note' };
    case 'reference':
      return inferContextSourceKindFromPath(path);
  }
}

export function classifyProjectContextSource(
  document: ProjectDocumentEntry,
  explicitKind: ContextSourceKind | null = null
): ContextSource | null {
  const kind = classifyContextSourceKind(document.document_type, document.path, explicitKind);

  if (!kind) {
    return null;
  }

  return {
    path: document.path,
    kind,
    // Phase 3.4 keeps discovered project sources opt-in until the app persists source-specific
    // activation preferences instead of silently defaulting whole categories into every task.
    activation_policy: 'explicit_only',
    review_state: 'user_authored'
  };
}

export function buildProjectContextSources(documents: ProjectDocumentEntry[]): ContextSource[] {
  return documents
    .map((document) => classifyProjectContextSource(document))
    .filter((source): source is ContextSource => source !== null);
}

function inferContextSourceKindFromPath(path: string): ContextSourceKind | null {
  const tokens = path
    .split('')
    .map((character) =>
      /[A-Za-z0-9]/.test(character) ? character.toLowerCase() : ' '
    )
    .join('')
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (tokens.length === 0) {
    return null;
  }

  if (hasAllTokens(tokens, ['prose']) && hasAnyToken(tokens, ['guide', 'guideline'])) {
    return { source_type: 'guide', source_kind: 'prose' };
  }

  if (hasAllTokens(tokens, ['style']) && hasAnyToken(tokens, ['guide', 'sheet'])) {
    return { source_type: 'guide', source_kind: 'style' };
  }

  if (hasAllTokens(tokens, ['critique']) && hasAnyToken(tokens, ['guide', 'rubric'])) {
    return { source_type: 'guide', source_kind: 'critique' };
  }

  if (hasAllTokens(tokens, ['rewrite']) && hasAnyToken(tokens, ['guide', 'brief'])) {
    return { source_type: 'guide', source_kind: 'rewrite' };
  }

  if (hasAllTokens(tokens, ['story']) && hasAnyToken(tokens, ['summary', 'synopsis'])) {
    return { source_type: 'reference', source_kind: 'story_summary' };
  }

  if (hasAllTokens(tokens, ['world']) && hasAnyToken(tokens, ['summary', 'overview'])) {
    return { source_type: 'reference', source_kind: 'world_summary' };
  }

  if (
    hasAnyToken(tokens, ['character', 'characters']) &&
    hasAnyToken(tokens, ['bible', 'sheet', 'sheets', 'guide', 'notes'])
  ) {
    return { source_type: 'reference', source_kind: 'character_bible' };
  }

  if (hasAnyToken(tokens, ['timeline', 'chronology'])) {
    return { source_type: 'reference', source_kind: 'timeline' };
  }

  if (hasAnyToken(tokens, ['terminology', 'glossary', 'lexicon', 'terms'])) {
    return { source_type: 'reference', source_kind: 'terminology' };
  }

  if (hasAnyToken(tokens, ['research', 'sourcebook'])) {
    return { source_type: 'reference', source_kind: 'research' };
  }

  return null;
}

function hasAnyToken(tokens: string[], expected: string[]): boolean {
  return expected.some((candidate) => tokens.includes(candidate));
}

function hasAllTokens(tokens: string[], expected: string[]): boolean {
  return expected.every((candidate) => tokens.includes(candidate));
}
