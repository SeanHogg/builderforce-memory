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
    rankEvermindRecall,
    hashRecallQuery,
    countReconciledMemories,
} from './recall.js';
export type { RecallScorable, RankedEvermindRecall } from './recall.js';
export { formatEvermindMemoryBlock } from './memoryBlock.js';

// Serving: text-quality gate, tool calling by constrained decoding, and the model-driving
// runtime (structural model/tokenizer shapes — still dependency-free).
export {
    EVERMIND_ANSWER_MIN_CHARS,
    assessTextCoherence,
    looksLikeCoherentText,
    isServableText,
} from './textCoherence.js';
export type { CoherenceFailure, CoherenceVerdict, CoherenceOptions } from './textCoherence.js';
export {
    TOOL_CHOICE_MIN_MARGIN,
    resolveToolChoiceMinMargin,
    normalizeEvermindTools,
    resolveEvermindToolChoice,
    renderToolsPreamble,
    planEvermindToolCall,
    toOpenAIToolCall,
} from './toolCall.js';
export type {
    NormalizedTool,
    JsonSchema,
    ToolChoicePlan,
    EvermindToolDecoder,
    EvermindPlannedCall,
    EvermindToolPlan,
} from './toolCall.js';
export {
    messagesToPrompt,
    generateEvermindText,
    COHERENCE_PROBE_PROMPTS,
    assessLMCoherence,
    probeEvermindText,
    createEvermindToolDecoder,
    generateEvermindWithTools,
} from './runtime.js';
export type {
    EvermindTextModel,
    EvermindTextTokenizer,
    EvermindUsage,
    EvermindGenerateOptions,
    EvermindGeneration,
    EvermindCoherenceSample,
    EvermindCoherenceAssessment,
    EvermindProbeGeneration,
    MeteredToolDecoder,
    EvermindToolGeneration,
} from './runtime.js';
