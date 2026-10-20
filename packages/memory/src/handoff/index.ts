/**
 * Model hand-off packing — the `@seanhogg/builderforce-memory/handoff` subpath (PRD 34 §3.3).
 *
 * When a conversation moves to another model (a local one with a small window, say),
 * {@link packHandoff} fits the transcript into that model's budget: the leading system
 * message(s) stay first, the latest user directive stays verbatim, tool calls are never
 * split from their results, the verbatim tail takes what the budget allows, and the
 * dropped middle becomes ONE note (summary → memory recall → elision marker, with a
 * one-line hand-off note when the models differ) whose role keeps turns alternating.
 *
 * Pure: no WebGPU, no IndexedDB, no `node:*` — runs in a Cloudflare Worker, a browser
 * page and Node alike.
 */

export { packHandoff } from './pack.js';
export { handoffBudget, defaultEstimateTokens, messageTokens, tokensOf } from './budget.js';
export { handoffLine, elisionMarker } from './note.js';
export type {
  HandoffMessage,
  HandoffPackInput,
  HandoffPack,
  HandoffBudgetInput,
  TokenEstimator,
} from './types.js';
