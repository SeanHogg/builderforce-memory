/**
 * Each distinct item with how often it occurs, most frequent first, ties by name — the
 * one ranking the brief, the consolidation and its area all use (and that hosts reuse
 * for their own renderings of a run, so a run reads the same everywhere).
 */
export function rankByCount(items: readonly string[]): [string, number][] {
  const counts = new Map<string, number>();
  for (const i of items) counts.set(i, (counts.get(i) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}
