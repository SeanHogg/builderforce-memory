/**
 * SSM.js – JavaScript-native AI runtime.
 *
 * Layer stack:
 *   MambaCode.js  →  WebGPU kernels (WGSL, Mamba-1/2/3 SSM math)
 *   SSM.js        →  Session layer + Runtime orchestration (this package)
 *
 * Quick start:
 *   import { SSM, AnthropicBridge } from 'ssmjs';
 *
 *   const ai = await SSM.create({
 *     session    : { modelSize: 'small' },
 *     bridge     : new AnthropicBridge({ apiKey: '...' }),
 *   });
 *
 *   await ai.adapt(myDocs);
 *   const answer = await ai.generate('How does MambaKit work?');
 */

// ── Session layer ─────────────────────────────────────────────────────────────
export { MambaSession }              from './session/index.js';
export { SessionError }              from './session/index.js';
export { MODEL_PRESETS, resolveLayerSchedule, resolveModelConfig } from './session/index.js';

export type { SessionErrorCode }     from './session/index.js';
export type { LayerSchedulePreset }  from './session/index.js';
export type {
    MambaSessionOptions,
    CompleteOptions,
    AdaptOptions,
    AdaptResult,
    SaveOptions,
    LoadOptions,
    StorageTarget,
    CreateProgressEvent,
    CreateStage,
    CreateCallbacks,
    SessionInternals,
    GpuMode,
    Tokenizer,
} from './session/index.js';

// ── Limbic system (trainable affective dynamics) ──────────────────────────────
export { LimbicSession } from './limbic/LimbicSession.js';
export type { LimbicSessionOptions, LimbicGpuMode } from './limbic/LimbicSession.js';
// Re-export the limbic engine primitives so consumers can use the model/trainer
// and the region schema directly from @seanhogg/builderforce-memory.
export {
    LimbicModel,
    LimbicTrainer,
    LIMBIC_DIM,
    LIMBIC_DIM_NAMES,
    LIMBIC_STATE_DIM,
    LIMBIC_BOUNDS,
    NEUTRAL_STATE,
    REGION,
    clampState,
    neutralState,
    stateToRecord,
    recordToState,
} from '@seanhogg/builderforce-memory-engine';
export type {
    LimbicModelConfig,
    LimbicForward,
    LimbicSample,
    LimbicTrainOptions,
    LimbicDimName,
    Region,
} from '@seanhogg/builderforce-memory-engine';

// ── Runtime ───────────────────────────────────────────────────────────────────
export { SSMRuntime }    from './runtime/SSMRuntime.js';
export type { SSMRuntimeOptions, GenerateOptions } from './runtime/SSMRuntime.js';

// ── Bridges ───────────────────────────────────────────────────────────────────
export type { TransformerBridge, BridgeGenerateOptions } from './bridges/TransformerBridge.js';
export { OpenAIBridge }    from './bridges/OpenAIBridge.js';
export { AnthropicBridge } from './bridges/AnthropicBridge.js';
export { FetchBridge }     from './bridges/FetchBridge.js';
export { CachingBridge }   from './bridges/CachingBridge.js';
export { SemanticCachingBridge } from './bridges/SemanticCachingBridge.js';
export { ResponseCache, buildCacheKey } from './bridges/ResponseCache.js';
export type { OpenAIBridgeOptions }    from './bridges/OpenAIBridge.js';
export type { AnthropicBridgeOptions } from './bridges/AnthropicBridge.js';
export type { FetchBridgeOptions }     from './bridges/FetchBridge.js';
export type { CachingBridgeOptions }   from './bridges/CachingBridge.js';
export type { SemanticCachingBridgeOptions } from './bridges/SemanticCachingBridge.js';
export type { ResponseCacheOptions }   from './bridges/ResponseCache.js';

// ── Semantic cache (embedding-keyed, L1 local + L2 shared) ─────────────────────
export { SemanticCache } from './cache/SemanticCache.js';
export { FetchSemanticCacheBackend } from './cache/FetchSemanticCacheBackend.js';
export type {
    Embedder,
    SemanticCacheBackend,
    SemanticCacheHit,
    SemanticCacheOptions,
    FetchSemanticCacheBackendOptions,
} from './cache/index.js';

// ── Similarity primitives ──────────────────────────────────────────────────────
export { cosineSimilarity, jaccardSimilarity, tokenize } from './similarity/index.js';

// ── Router ────────────────────────────────────────────────────────────────────
export { InferenceRouter } from './router/InferenceRouter.js';
export type {
    RoutingStrategy,
    RoutingDecision,
    RouterContext,
    InferenceRouterOptions,
    RoutingAuditEntry,
} from './router/InferenceRouter.js';

// ── Memory ────────────────────────────────────────────────────────────────────
export { MemoryStore }  from './memory/MemoryStore.js';
export type {
    MemoryEntry,
    MemoryStoreOptions,
    RememberOptions,
    FactType,
} from './memory/MemoryStore.js';

// ── Cognition (Evermind — Write-Through Cognition) ────────────────────────────
export { EvermindCognition, workspacePresenceGatherer } from './cognition/index.js';
export type {
    EvermindCognitionOptions,
    WorkspacePresenceRule,
    Claim,
    CognitionFactStore,
    CommitResult,
    EvidenceContext,
    EvidenceGatherer,
    EvidenceResult,
    Verdict,
} from './cognition/index.js';

// ── Distillation ──────────────────────────────────────────────────────────────
export { DistillationEngine } from './distillation/DistillationEngine.js';
export type {
    DistillOptions,
    DistillResult,
    DistillBatchResult,
    DistillationLog,
    QualityGate,
} from './distillation/DistillationEngine.js';

// ── Agent ─────────────────────────────────────────────────────────────────────
export { SSMAgent }  from './agent/SSMAgent.js';
export type { SSMAgentOptions, ThinkOptions, AgentMessage, MessageRole } from './agent/SSMAgent.js';

// ── Errors ────────────────────────────────────────────────────────────────────
export { SSMError }  from './errors/SSMError.js';
export type { SSMErrorCode } from './errors/SSMError.js';

// ── Top-level SSM namespace ───────────────────────────────────────────────────
// Allows the `SSM.create()` pattern from the spec:
//   const ai = await SSM.create({ session: { modelSize: 'nano' } });

import { SSMRuntime }          from './runtime/SSMRuntime.js';
import type { SSMRuntimeOptions } from './runtime/SSMRuntime.js';

export const SSM = {
    /**
     * Creates a new SSMRuntime.
     *
     * Shorthand for `SSMRuntime.create(opts)`.
     * Can throw `SessionError` for GPU / tokenizer failures during init.
     */
    create: (opts: SSMRuntimeOptions) => SSMRuntime.create(opts),
} as const;
