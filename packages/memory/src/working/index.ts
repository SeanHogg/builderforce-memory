/**
 * Working memory — the `@seanhogg/builderforce-memory/working` subpath.
 *
 * A run has two memories, and they do different jobs:
 *
 *   - SHORT-TERM: {@link WorkingMemory}, what THIS run has read, searched and changed.
 *     Restated before every turn ({@link WorkingMemory.brief}), because a model's own
 *     reasoning is not carried into its next turn and old tool results are compacted
 *     out of its window — without it a long run re-derives the codebase every turn.
 *   - LONG-TERM: {@link consolidate}, what survives the run — a map of where the code
 *     for that area lives, keyed per area so the next run on it (any surface) opens the
 *     right files first. {@link mergeConsolidated} is the write-side rule that makes
 *     consolidations of one area accumulate instead of overwriting each other.
 *
 * The memory knows observations ("read this file", "changed that one"), never tool
 * names — a host maps its own tool vocabulary onto {@link Observation}.
 *
 * Pure: no `node:*`, no I/O — runs in a Cloudflare Worker, a browser page and Node alike.
 */

export { WorkingMemory } from './workingMemory.js';
export { consolidate, mergeConsolidated, consolidatedKey, CONSOLIDATED_KEY_PREFIX, MIN_CONSOLIDATED_FILES } from './consolidate.js';
export { rankByCount } from './rank.js';
export type { Observation, WorkingMemorySnapshot, ConsolidatedMemory, ConsolidateOptions } from './types.js';
