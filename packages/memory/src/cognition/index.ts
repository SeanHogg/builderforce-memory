/**
 * Evermind — Write-Through Cognition.
 *
 * The layer that keeps model knowledge current without a reconciliation step:
 * stable-subject-key beliefs, evidence-gated conflict resolution, replace-on-write.
 */

export { EvermindCognition } from './EvermindCognition.js';
export type { EvermindCognitionOptions } from './EvermindCognition.js';
export { workspacePresenceGatherer } from './gatherers.js';
export type { WorkspacePresenceRule } from './gatherers.js';
export {
    canonicalizeSubjectKey,
    normalizeSubjectKey,
    buildAliasTable,
} from './canonicalize.js';
export type { AliasTable } from './canonicalize.js';
export {
    sanitizeRecalledFact,
    buildRecallContext,
    trustScore,
} from './sanitize.js';
export type { RecalledFact, FactProvenance } from './sanitize.js';
export type {
    Claim,
    CognitionFactStore,
    CommitResult,
    EvidenceContext,
    EvidenceGatherer,
    EvidenceResult,
    Verdict,
} from './types.js';
