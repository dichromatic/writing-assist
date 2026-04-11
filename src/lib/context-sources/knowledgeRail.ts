import type { ContextSource } from '$lib/task/types';

export type KnowledgeRailState = {
  availableSources: ContextSource[];
  activeContextPaths: string[];
};

export function createKnowledgeRailState(availableSources: ContextSource[]): KnowledgeRailState {
  return {
    availableSources,
    activeContextPaths: []
  };
}

export function setKnowledgeRailActivePaths(
  state: KnowledgeRailState,
  nextPaths: string[]
): KnowledgeRailState {
  const allowedPaths = new Set(state.availableSources.map((source) => source.path));
  const activeContextPaths = Array.from(new Set(nextPaths)).filter((path) => allowedPaths.has(path));

  return {
    ...state,
    activeContextPaths
  };
}

export function toggleKnowledgeRailPath(
  state: KnowledgeRailState,
  path: string
): KnowledgeRailState {
  const activePaths = new Set(state.activeContextPaths);

  if (activePaths.has(path)) {
    activePaths.delete(path);
  } else if (state.availableSources.some((source) => source.path === path)) {
    activePaths.add(path);
  }

  return {
    ...state,
    activeContextPaths: Array.from(activePaths)
  };
}
