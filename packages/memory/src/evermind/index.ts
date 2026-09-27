/**
 * Evermind — the runtime half of the Evermind module (`@seanhogg/builderforce-memory/evermind`):
 * the project-Evermind recall contract, the lexical recall and reconcile rules, and the
 * injection-safe memory block. Dependency-free, so a browser chat client can import it
 * without pulling the rest of the runtime.
 *
 * The engine half (the `.evermind` artifact, adapt, delta wire, merge, eval) is in
 * `@seanhogg/builderforce-memory-engine`; Write-Through Cognition is `EvermindCognition`.
 */
export type { EvermindRecallItem, EvermindRecallResult } from './contract.js';
export { EVERMIND_MIN_TEACH_CHARS } from './contract.js';
export {
    RECALL_STOPWORDS,
    RECONCILE_OVERLAP,
    recallTokens,
    rankEvermindRecall,
    hashRecallQuery,
    countReconciledMemories,
} from './recall.js';
export type { RecallScorable, RankedEvermindRecall } from './recall.js';
export { formatEvermindMemoryBlock } from './memoryBlock.js';
